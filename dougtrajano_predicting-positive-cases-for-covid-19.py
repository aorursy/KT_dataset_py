import numpy as np

import pandas as pd

import json

import pandas_profiling as pdp

import missingno

from tqdm import tqdm

from IPython.core.display import HTML, Image

import os



# graphs

import seaborn as sns

import matplotlib.pyplot as plt



# ML

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.impute import KNNImputer



from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
# Variables

pandas_profiling_executor = False # Change it for True if you want to update the latest report

pandas_profiling_file = "covid-19_br_einsteindata4u.html"

pandas_profiling_link = "https://github.com/DougTrajano/ds_covid-19_brazil/blob/master/covid-19_br_einsteindata4u.html"

target_col = "SARS-Cov-2 exam result"

file_path = "/kaggle/input/covid19/dataset.xlsx"
print(file_path)

df = pd.read_excel(file_path)



print(df.shape)

df.head()
if pandas_profiling_executor:

    profile = pdp.ProfileReport(df, title='COVID-19 - Hospital Albert Einstein BR')

    profile.to_file(output_file=pandas_profiling_file)

    pandas_profiling_link = pandas_profiling_file

    print("pandas_profiling_link changed to {}".format(pandas_profiling_file))

    

HTML('<a href="{href}" target="_blank">{link_name}</a>'.format(href=pandas_profiling_link,

                                                               link_name=pandas_profiling_file))
cols_convert_to_binary = ["SARS-Cov-2 exam result", "Respiratory Syncytial Virus", "Influenza A", "Influenza B",

                         "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus", "Coronavirus HKU1",

                         "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus", "Parainfluenza 4",

                         "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", 

                         "Metapneumovirus", "Parainfluenza 2", "Influenza B, rapid test", "Influenza A, rapid test",

                         "Strepto A", "Urine - Esterase", "Urine - Hemoglobin", "Urine - Bile pigments",

                         "Urine - Ketone Bodies", "Urine - Urobilinogen", "Urine - Protein", "Urine - Hyaline cylinders",

                         "Urine - Granular cylinders", "Urine - Yeasts", ]



cols_categorical = ["Urine - Aspect", "Urine - Crystals", "Urine - Leukocytes", "Urine - Color"]



cols_invalid = ["Patient ID", "Mycoplasma pneumoniae", "Fio2 (venous blood gas analysis)", "Urine - pH",

                "Urine - Nitrite", "Urine - Sugar", "Partial thromboplastin time\xa0(PTT)\xa0", "Vitamin B12"]
missingno.matrix(df);
df = df[df["Hemoglobin"].notnull()]

df.reset_index(drop=True, inplace=True)

print(df.shape)
df_missing = []

for col in df.columns:

    missing = (len(df[df[col].notnull()]) / len(df))*100

    missing = round(abs(missing - 100),2)

    df_missing.append({"column": col, "missing percentage": missing})

    if missing == 0:

        print("Column {col} has no missing values.".format(col=col))

    else:

        print("Column {col} has {missing}% of missing values.".format(col=col, missing=missing))



df_missing = pd.DataFrame(df_missing)
missingno.matrix(df);
def _convert_to_binary(value):

    positive_lst = [1, "positive", "detected", "present", "normal"]

    negative_lst = [0, "negative", "not_detected", "absent", "not_done"]

    if value in positive_lst:

        return 1

    elif value in negative_lst:

        return 0

    else:

        return value

    

    

def processing(dataset, cols_invalid=None, cols_categorical=None, cols_convert_to_binary=None):

    """

    docstring

    """

    temp = dataset.to_dict(orient="records")    

    df_processed = []

    

    # processing each record

    with tqdm(total=len(temp)) as pbar:

        for values in temp:

            if isinstance(cols_convert_to_binary, list):

                for col in cols_convert_to_binary:

                    values[col] = _convert_to_binary(values[col])



            if isinstance(cols_invalid, list):    

                for col in cols_invalid:

                    del values[col]



            if isinstance(cols_categorical, list):

                for col in cols_categorical:

                    values = _encoder(values)

        

            # add processed record

            df_processed.append(values)

            pbar.update(1)

        

    df_processed = pd.DataFrame(df_processed)

    return df_processed





def _encoder(value):

    filename = "cat_features_encoding.json"

    with open(filename, 'r') as filename:

        encoding = json.load(filename)

        

    for col in encoding.keys():        

        for i in encoding[col]:

            if value[col] == i:

                value[col] = encoding[col][i]             

        

    return value





def create_encoder(dataset, cat_features):

    """

    This function can create a Label Encoder for categorical features.

    

    A json file called "cat_features_encoding.json" will be saved on folder's script. This file can be used on _encoder function.

    Input

    - dataset (DataFrame, required): The DataFrame loaded from listings.csv.

    - cat_features (list, required): The categorical features list that you want to convert in numeric values.

    Output

    - encoding (dict): A dictionary with the encoder created. The same content of json file.

    """

    encoding = {}

    filename = "cat_features_encoding.json"

    for col in cat_features:

        temp = {}

        n_values = dataset[col].unique()

        i = 0

        for n in n_values:

            temp[str(n)] = i

            i += 1

        encoding[col] = temp

    

    # save json file encoder

    with open(filename, 'w') as filename:

        json.dump(encoding, filename)

        

    return encoding
for col in cols_categorical:

    print(col, len(df[col].unique()))

    print(df[col].unique())

    print()
create_encoder(df, cols_categorical)
df_processed = processing(df, cols_invalid=cols_invalid, cols_categorical=cols_categorical,

                         cols_convert_to_binary=cols_convert_to_binary)



print(df_processed.shape)

df_processed.head()
df_processed.info()
# Correlation features

cor = df_processed.corr()

cor_target = abs(cor[target_col])



#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.05].sort_values(ascending=False)



df_cols_stats = pd.DataFrame(relevant_features[1:])

df_cols_stats.reset_index(inplace=True)

df_cols_stats.columns = ["column", "correlation with exam result"]

df_cols_stats = pd.merge(df_cols_stats, df_missing, on="column")

df_cols_stats.head(20)
cols_pairplot_filter = df_cols_stats[df_cols_stats["missing percentage"] < 90]["column"].tolist()

cols_pairplot_filter.append(target_col) # add target_col to the columns



df_cols_stats[df_cols_stats["missing percentage"] < 90]
df_processed[cols_pairplot_filter].groupby(by=target_col).count().transpose()
%%time

pairplot_file = "pairplot_1.png"

temp = cols_pairplot_filter[:20]

if target_col not in temp:

    temp.append(target_col)



print("Pairplot - 1")

sns.pairplot(df_processed[temp], hue=target_col, diag_kind="hist", dropna=True)

plt.savefig(pairplot_file)



plt.clf() # Clean parirplot figure from sns



Image(filename=pairplot_file) # Show pairplot as image
# Download image

HTML('<a href="{href}" target="_blank">Download image (1)</a>'.format(href=pairplot_file))
%%time

pairplot_file = "pairplot_2.png"

temp = cols_pairplot_filter[20:]

if target_col not in temp:

    temp.append(target_col)

    

print("Pairplot - 2")

sns.pairplot(df_processed[temp], hue=target_col, diag_kind="hist", dropna=True)

plt.savefig(pairplot_file)



plt.clf() # Clean parirplot figure from sns



Image(filename=pairplot_file) # Show pairplot as image
# Download image

HTML('<a href="{href}" target="_blank">Download image (2)</a>'.format(href=pairplot_file))
df_processed[cols_pairplot_filter].info()
X = df_processed[cols_pairplot_filter]



imputer = KNNImputer(n_neighbors=4)

X = imputer.fit_transform(X.values)



X = pd.DataFrame(X, columns=cols_pairplot_filter)

X.head()
y = X[target_col].values

X.drop(columns=[target_col], inplace=True)
temp = pd.DataFrame(df_processed[target_col].value_counts())



for each in range(len(temp)):

    percentage = temp.values[each]/sum(temp.values)

    percentage = round(percentage[0]*100, 2)

    print("Class {ix}: {qty} - ({percentage}%)".format(ix=temp.index[each], qty=temp.values[each], percentage=percentage))



df_processed["SARS-Cov-2 exam result"].value_counts().plot(kind="bar", title="Classes distribution", legend=True, rot=1);
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)



class_weights = {

    0: class_weights[0],

    1: class_weights[1]

}



print(class_weights)
# Split the dataset in train and test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



print("train size:", len(X_train))

print("test size:", len(X_test))
%%time

clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0, 

                             min_samples_leaf=5, min_samples_split=15,

                             class_weight=class_weights, n_jobs=-1)



clf.fit(X_train, y_train)



scores = cross_val_score(clf, X, y, cv=5)

y_pred = clf.predict(X_test)



print("Accuracy (cross-validation): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print()

print(classification_report(y_test, y_pred, digits=4))
%%time



parameters = {'n_estimators':[25, 50, 100, 200, 300],

             "min_samples_leaf": [5, 10, 15, 20],

             "min_samples_split": [5, 10, 15, 20],

             "criterion": ["gini", "entropy"]}



model = RandomForestClassifier(random_state=0, class_weight=class_weights, n_jobs=-1)

clf = GridSearchCV(model, parameters, cv=5, verbose=2, n_jobs=-1)

clf.fit(X_train, y_train)

clf = clf.best_estimator_

preds = clf.predict(X_test)

print(classification_report(y_test, preds, digits=4))

clf