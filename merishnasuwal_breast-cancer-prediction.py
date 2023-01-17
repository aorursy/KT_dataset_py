import pandas as pd 
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
%matplotlib inline
df = pd.read_csv("../input/Breast_cancer_data.csv", delimiter=",")
df.head() #gives first 5 entries of a dataframe by default
df.columns
df.isnull().sum()
count = df.diagnosis.value_counts()
count
count.plot(kind='bar')
plt.title("Distribution of malignant(1) and benign(0) tumor")
plt.xlabel("Diagnosis")
plt.ylabel("count");
y_target = df['diagnosis']
df.columns.values
df['target'] = df['diagnosis'].map({0:'B',1:'M'}) # converting the data into categorical
g = sns.pairplot(df.drop('diagnosis', axis = 1), hue="target", palette='prism');
sns.scatterplot(x='mean_perimeter', y = 'mean_texture', data = df, hue = 'target', palette='prism');
features = ['mean_perimeter', 'mean_texture']
X_feature = df[features]
# X_feature = df.drop(['target','diagnosis'], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_feature, y_target, test_size=0.3, random_state = 42)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(X_train, y_train)
from mlxtend.plotting import plot_decision_regions
# !pip install mlxtend
plot_decision_regions(X_train.values, y_train.values, clf=model, legend=2)
plt.title("Decision boundary for Logistic Regression (Train)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture");
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy score using Logistic Regression:", acc*100)
plot_decision_regions(X_test.values, y_test.values, clf=model, legend=2)
plt.title("Decision boundary for Logistic Regression (Test)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture");
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy score using KNN:", acc*100)
confusion_matrix(y_test, y_pred)
plot_decision_regions(X_train.values, y_train.values, clf=clf, legend=2)
plt.title("Decision boundary using KNN (Train)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture");
plot_decision_regions(X_test.values, y_test.values, clf=clf, legend=2)
plt.title("Decision boundary using KNN (Test)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture");