import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.chained_assignment = None

from sklearn import preprocessing, model_selection, svm, ensemble, linear_model

import datetime

import csv



train_df = pd.read_csv("../input/train.csv")

train_df.fillna(0,  inplace=True)

predict_df = pd.read_csv("../input/test.csv")

predict_df.fillna(0,  inplace=True)



SVC_FIELDS = ["OverallQual", "OverallCond", "BedroomAbvGr", "FullBath", "HalfBath", "KitchenAbvGr",

                 "BsmtFullBath", "BsmtHalfBath", "TotRmsAbvGrd", "GarageCars", "Fireplaces",

                 "lot_area_percentage", "lot_frontage_percentage", "Age", "LastReno"]

NORMALIZATION_FIELDS = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities",

                        "LotConfig", "LandSlope", "Condition1", "Condition2", "BldgType",

                        "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",

                        "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual",

                        "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",

                        "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional",

                        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",

                        "PavedDrive", "SaleType", "SaleCondition"]



# def neighborhood_increase_pa():

#     increases = {}

#     for neighborhood in set(list(train_df["Neighborhood"])):

#         yearly_increases = {}

#         filtered_neighborhood = train_df[train_df.Neighborhood == neighborhood]

#         for year in set(list(train_df["YrSold"])):

#             filtered_year = filtered_neighborhood[filtered_neighborhood.YrSold == year]

#             yearly_increases[year] = filtered_year["SalePrice"].mean()

#         increases[neighborhood] = yearly_increases

#     return(increases)



def normalize_field(field):

    normalization = {field: {}}

    i = 0

    for row in set(list(train_df[field]) + [0]):

        normalization[field][row] = i

        i += 1

    return(normalization)



def normalized_neighborhoods():

    neighborhoods = {}

    for neighborhood in set(list(train_df["Neighborhood"])):

        df = train_df[train_df.Neighborhood == neighborhood]

        p_df = predict_df[predict_df.Neighborhood == neighborhood]

        neighborhoods[neighborhood] = {"df": df, "predict_df": p_df, "mean": df["SalePrice"].mean(), "std": df["SalePrice"].std(),

                                      "LotFrontageMax": df["LotFrontage"].max(), "LotAreaMax": df["LotArea"].max()}

    return(neighborhoods)



def apply_saleprice_stdev(df, values, i):

    stddev = int(round((df["SalePrice"][i]-values["mean"])/values["std"]))

    df.loc[i,'stddev'] = stddev



def apply_lot_sizes(df, values, i):

    lot_area = int((df["LotArea"][i]/values["LotAreaMax"])*10)

    df.loc[i,'lot_area_percentage'] = lot_area

    lot_frontage = int((df["LotFrontage"][i]/values["LotFrontageMax"])*10)

    df.loc[i,'lot_frontage_percentage'] = lot_frontage



def apply_years_old_and_reno(df, values, i):

    df.loc[i, "Age"] = int(datetime.datetime.now().year) - int(df["YearBuilt"][i])

    df.loc[i, "LastReno"] = int(datetime.datetime.now().year) - int(df["YearRemodAdd"][i])



def apply_field_norm(df, field, i, normalization):

    df.loc[i, field] = normalization[field][df[field][i]]



def apply_normalization(df, apply_to, values, normalization):

    for i in df.index:

        if apply_to: apply_saleprice_stdev(df, values, i)

        apply_lot_sizes(df, values, i)

        apply_years_old_and_reno(df, values, i)

        [apply_field_norm(df, field, i, normalization) for field in NORMALIZATION_FIELDS]



def train_std(df):

    X = np.array(df[SVC_FIELDS + NORMALIZATION_FIELDS])

    y_lr = np.array(df["SalePrice"])

    y = np.array(df["stddev"])



    X_train_lr, X_test_lr, y_train_lr, y_test_lr = model_selection.train_test_split(X, y_lr, test_size=0.2)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)



    LR_clf = linear_model.LinearRegression()

    SVM_clf = svm.SVC()

    RF_clf = ensemble.RandomForestClassifier()



    LR_clf.fit(X_train_lr, y_train_lr)

    SVM_clf.fit(X_train, y_train)

    RF_clf.fit(X_train, y_train)



    print("\nlinear Regression: {}%".format(round(LR_clf.score(X_test_lr, y_test_lr)*100)))

    print("Support Vector Machine: {}%".format(round(SVM_clf.score(X_test, y_test)*100)))

    print("Random Forest: {}%\n".format(round(RF_clf.score(X_test, y_test)*100)))



    clfs = [LR_clf, SVM_clf, RF_clf]

    return(clfs)



def predict_std(predict_df, clfs, mean, std):

    predict_X = np.array(predict_df[SVC_FIELDS + NORMALIZATION_FIELDS])

    predict_y = np.array(predict_df[["Id"]])

    i = 0

    predictions = []

    for row in predict_X:

        row = row.reshape(1, -1)

        LR_prediction = clfs[0].predict(row)

        SVM_prediction = clfs[1].predict(row)

        RF_prediction = clfs[2].predict(row)



        SVM_prediction = (SVM_prediction[0] * std) + mean

        RF_prediction = (RF_prediction[0] * std) + mean



        prediction = (LR_prediction[0] + SVM_prediction + RF_prediction) / 3

        predictions.append([predict_y[i][0], prediction])

        i += 1



    return(predictions)



def save_predictions(predictions):

    with open('predictions.csv', 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(["Id", "SalePrice"])

        for row in predictions:

            writer.writerow(row)

    print("Predictions Saved.")



def process():

    train_dfs = []

    normalization = {}

    [normalization.update(normalize_field(field)) for field in NORMALIZATION_FIELDS]

    neighborhoods = normalized_neighborhoods()

    for neighborhood, values in neighborhoods.items():

        apply_normalization(values["df"], True, values, normalization)

        train_dfs.append(values["df"])

        print(neighborhood + " Complete.")

    clfs = train_std(pd.concat(train_dfs))



    # Number of Classes based on STDDEV

    # print(len(set(list(pd.concat(train_dfs)["stddev"]))))



    predictions = []

    for neighborhood, values in neighborhoods.items():

        apply_normalization(values["predict_df"], False, values, normalization)

        predictions.extend(predict_std(values["predict_df"], clfs, values["mean"], values["std"]))

    save_predictions(predictions)



process()