import pandas as pd
import seaborn as sns
import sklearn.model_selection as ms
import sklearn.neighbors as nb
import sklearn.metrics as mt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/Iris.csv')
df.head()
sns.lmplot(x='SepalLengthCm', y='SepalWidthCm',data=df,fit_reg=False)
sns.lmplot(x='PetalLengthCm', y='PetalWidthCm',data=df,fit_reg=False)
features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
target = 'Species'

train_data, test_data = ms.train_test_split(df, test_size = 0.2)
len(test_data)
classifier = nb.KNeighborsClassifier()

classifier.fit(train_data[features], train_data[target])
predictions = classifier.predict(test_data[features])

print(mt.classification_report(test_data[target], predictions))
