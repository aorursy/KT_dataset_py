import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
!cd ../input && ls
db = pd.read_csv('../input/nbclassifier/train_data.csv')
db.head()
db.shape
db.ham.value_counts()
count_Class=pd.value_counts(db["ham"], sort= True)
count_Class.plot(kind= 'bar', color= ["blue", "green"])
plt.title('Ham or spam count')
plt.show()
ham_data = db.loc[(db['ham']==True)]
ham_data.head()
ham_data_top10 = ham_data[(ham_data.iloc[:,0:54].select_dtypes(include=['number']) >20).any(1)]
ham_data_top10
ham_column_labels=[]
count_ham = 0
for column in ham_data.iloc[:,0:54].columns:
    dif_zero = ham_data[column].unique()
    if len(dif_zero)>1:
        for freq in dif_zero:
            if all([freq>18, column not in ham_column_labels]):
                ham_column_labels.append(column)
                print(column + ": " + str(freq))
                count_ham=count_ham+1
print("\nLabels used: " + str(count_ham))
spam_data = db.loc[(db['ham']==False)]
spam_data.head()
spam_column_labels=[]
count = 0
for column in spam_data.iloc[:,0:54].columns:
    dif_zero = spam_data[column].unique()
    if len(dif_zero)>1:
#         print(column + str(dif_zero))
        for freq in dif_zero:
            if all([freq>8, column not in spam_column_labels]):
                spam_column_labels.append(column)
                print(column + ": " + str(freq))
                count=count+1
print("\nLabels used: " + str(count))
labels=[]
for spam_label in spam_column_labels:
    if spam_label not in ham_column_labels:
        labels.append(spam_label)
for ham_label in ham_column_labels:
    if ham_label not in spam_column_labels:
        labels.append(ham_label)
labels
plt.figure(None,figsize=(13,9))
plt.subplots_adjust(top=2)

for i,label in enumerate(labels):

    mean_ham = np.mean(ham_data[label])
    mean_spam = np.mean(spam_data[label])
    heights = [mean_ham,mean_spam]

    locations = [1, 2]
    l = ["Ham", "Spam"]

    plt.subplot(6,3,i+1)
    plt.bar(locations, heights, tick_label=l)
    plt.title(label)
for i,label in enumerate(labels):
    if (label == 'char_freq_(' or label == 'word_freq_mail'):
        labels.pop(i)
labels
X_values = db[labels]
Y_values = db['ham']
prediction = []
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(X_values,Y_values)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_values,Y_values, cv=10)
scores
test_db = pd.read_csv('../input/nbclassifier/test_features.csv')
test_db.head()
test_db.shape
prediction = dict()
prediction["ham"] = model.predict(test_db[labels])
Id = test_db['Id']
pred_df = pd.DataFrame(prediction, Id)
pred_df.to_csv('predictions2.csv')
#pred_df = pred_df.drop(pred_df.columns[[0]], axis=1)
pred_df.head()
