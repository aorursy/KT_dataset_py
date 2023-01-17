import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.svm import SVC



from sklearn.model_selection import train_test_split



import seaborn as sns
data = pd.read_csv('../input/diamond/diamonds_filter.csv')

data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data.cut = le.fit_transform(data.cut)

data
data1 = data.drop(['cut'], axis='columns') # Independent variable

data2 = data.cut  #Dependent Variable
data1.hist(grid=True, figsize=(20,10), color='darkred')
plt.figure(figsize=(20,10))

sns.pairplot(data1, diag_kind='kde')
sns.jointplot(x="carat", y="depth", data=data1, kind="kde")
sns.jointplot(x="table", y="clarity", data=data1, kind="kde")
x_train, x_test, y_train, y_test = train_test_split(data1,data2, test_size=0.3, random_state=0) 
model = SVC()
model.fit(x_train, y_train)
model.score(x_test, y_test)
model.predict([['0.50', '66.5', '58.0', '4.0']])
model_linear_kernel = SVC(kernel = 'linear')

model_linear_kernel.fit(x_train, y_train)

model_linear_kernel.score(x_test, y_test)
model_C = SVC(C=4)

model_linear_kernel.fit(x_train, y_train)

model_linear_kernel.score(x_test, y_test)
model_g = SVC(gamma=6)

model_linear_kernel.fit(x_train, y_train)

model_linear_kernel.score(x_test, y_test)