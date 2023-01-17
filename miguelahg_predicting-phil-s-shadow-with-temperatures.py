import numpy as np



import matplotlib.pyplot as plt



import pandas as pd

from pandas.plotting import scatter_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv("../input/groundhog-day/archive.csv")
print(df.shape)
print(df.count())
print(df.iloc[list(range(9)) + [131], :3])
print(df["Punxsutawney Phil"].value_counts())
df = df.dropna()



recorded_mask = df["Punxsutawney Phil"] != "No Record"

df = df[recorded_mask]



df = df.replace("Partial Shadow", "Full Shadow")



print(df.shape)

print(df.iloc[:, :3].head(10))
print(df.count())
print(df["Punxsutawney Phil"].value_counts())
scatter_matrix(df)
color_dict = {

    "No Shadow": 0,

    "Full Shadow": 1,

}



color_list = [color_dict[shadow] for shadow in df["Punxsutawney Phil"]]



feb_mar_scatter = plt.scatter(

    df["February Average Temperature"],

    df["March Average Temperature"],

    c = color_list,

)



plt.title("February Average Temperature and March Average Temperature, by Class of Shadow")

plt.xlabel("February Average Temperature")

plt.ylabel("March Average Temperature")

plt.legend(

    handles = feb_mar_scatter.legend_elements()[0],

    labels = color_dict.keys(),

)



plt.show()
X = df[["February Average Temperature", "March Average Temperature"]].values

y = (df["Punxsutawney Phil"] == "No Shadow").values



# Print the first 5 observations.

print(X[:5])

print(y[:5])
color_dict = {

    "No Shadow": 0,

    "Full Shadow": 1,

}



color_list = [color_dict[shadow] for shadow in df["Punxsutawney Phil"]]



feb_mar_scatter = plt.scatter(

    df["February Average Temperature"],

    df["March Average Temperature"],

    c = color_list,

)



plt.title("February Average Temperature and March Average Temperature, by Class of Shadow")

plt.xlabel("February Average Temperature")

plt.ylabel("March Average Temperature")

plt.legend(

    handles = feb_mar_scatter.legend_elements()[0],

    labels = color_dict.keys(),

)



# Manually add an approximated line that separates the two classes.

plt.plot([26, 42], [50, 36])



plt.show()
def score_model(X, y, kf):

    acc_scores = []

    prec_scores = []

    recall_scores = []

    f1_scores = []



    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]



        model = LogisticRegression()

        model.fit(X_train, y_train)



        y_pred = model.predict(X_test)

        y_set = (y_test, y_pred)

        

        acc_scores.append(accuracy_score(*y_set))

        prec_scores.append(precision_score(*y_set, zero_division = 0))

        recall_scores.append(recall_score(*y_set))

        f1_scores.append(f1_score(*y_set))

        

    scores_str = """Accuracy: {acc}

Precision: {prec}

Recall: {recall}

F1 Score: {f1}""".format(

        acc = np.mean(acc_scores),

        prec = np.mean(prec_scores),

        recall = np.mean(recall_scores),

        f1 = np.mean(f1_scores),

    )

    

    return scores_str
kf = KFold(

    n_splits = 5,

    shuffle = True,

)



print(score_model(X, y, kf))
model = LogisticRegression()

model.fit(X, y)
coef_1, coef_2 = model.coef_[0]

intercept = model.intercept_[0]



print("""{coef_1} * x1 + {coef_2} * x2 + {intercept} = y



Therefore, excluding y from the equation,



x2 = ({coef_1} * x1 + {intercept}) / -({coef_2})""".format(

    coef_1 = coef_1,

    coef_2 = coef_2,

    intercept = intercept,

))
color_dict = {

    "No Shadow": 0,

    "Full Shadow": 1,

}



color_list = [color_dict[shadow] for shadow in df["Punxsutawney Phil"]]



feb_mar_scatter = plt.scatter(

    df["February Average Temperature"],

    df["March Average Temperature"],

    c = color_list,

)



plt.title("February Average Temperature and March Average Temperature, by Class of Shadow")

plt.xlabel("February Average Temperature")

plt.ylabel("March Average Temperature")

plt.legend(

    handles = feb_mar_scatter.legend_elements()[0],

    labels = color_dict.keys(),

)



# Plot the logistic regression model's line.

x = np.linspace(25, 42.5, 1000)

y = (coef_1 * x + intercept) / -coef_2

plt.plot(x, y)



plt.show()