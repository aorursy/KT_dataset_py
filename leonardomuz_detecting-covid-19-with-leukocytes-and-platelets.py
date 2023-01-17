import pandas as pd

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv")
df.sars_cov_2_exam_result.replace('negative', 0, inplace=True)

df.sars_cov_2_exam_result.replace('positive', 1, inplace=True)
selected_columns = ['sars_cov_2_exam_result','platelets','leukocytes']



df = df[selected_columns]
df.dropna(inplace=True)

toTrain = df.drop('sars_cov_2_exam_result', axis=1).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(toTrain, df.sars_cov_2_exam_result, test_size=0.33, random_state=42)
gnb = GaussianNB()
gnb.fit(X_train,y_train)
preds = gnb.predict(X_test)

y_test = y_test.to_numpy()
truePositive = 0

trueNegative = 0

falsePositive = 0

falseNegative = 0



for i in range(len(preds)):

    if(preds[i]==y_test[i]):

        if(preds[i]==1):

            truePositive += 1

        else:

            trueNegative += 1

    else:

        if(preds[i]==1):

            falsePositive += 1

        else:

            falseNegative += 1



print("True Positive predictions:",truePositive)

print("True Negative predictions:",trueNegative)

print("False Positive predictions:",falsePositive)

print("False Negative predictions:",falseNegative)

print("Accuracy:", (truePositive+trueNegative)/(truePositive+trueNegative+falsePositive+falseNegative)*100)