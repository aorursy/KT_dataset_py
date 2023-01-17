import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
%matplotlib inline

# The determination whether to issue a warning message is controlled by the warning filter,
# which is a sequence of matching rules and actions. 
# Rules can be added to the filter by calling filterwarnings() and 
# reset to its default state by calling resetwarnings().
warnings.filterwarnings('ignore')

# color_codes: bool .If True and palette is a seaborn palette, remap the shorthand color codes (e.g. “b”, “g”, “r”, etc.) to the colors from this palette.
sns.set(style='white',color_codes=True)
sns.set_palette('Set1')
import io
df = pd.read_csv('/kaggle/input/personal-loan/Bank_Personal_Loan_Modelling-1.xlsx')
df.rename(columns = {'ZIP Code' : 'ZIP_Code', 'CCAvg' : 'CC_Avg', 'Personal Loan':'Personal_Loan', 'Securities Account' : 'Securities_Account', 'CD Account':'CD_Account', 'CreditCard' : 'Credit_Card'}, inplace=True) 
df.head()
df.shape
df.info()
df.isna().apply(pd.value_counts)
df.describe().T
sns.pairplot(df)
corr = df.corr()

plt.figure(figsize=(12,10))
# y float: Vertical axes loation for the title (1.0 is the top).
plt.title('Pearson correlation of Attributes', y = 1, size = 19)
sns.heatmap(corr, cmap = 'YlGnBu', annot=True, linewidths=.5, fmt= '.2f', center = 1) # fmtstring: String formatting code to use when adding annotations.
# other cmap options: 'Blues', 'Greens', 'BuPu'
print("Negative Experience Count:", df[df['Experience'] < 0]['Experience'].count())
df.drop(['Experience','ID'],axis=1,inplace=True)
df.columns
sns.boxplot(df['ZIP_Code'])
print(df[df['ZIP_Code']<20000])
# Lets drop that row as ZIP code 9307 is not a valid US zip code.
df.drop(index = 384, inplace=True)
plt.figure(figsize=(20,7))
sns.scatterplot(x = df['ZIP_Code'], y = df['Income'], hue = df['Personal_Loan'], alpha = 0.5)
print("Checking concentration of Income in ZIP codes and relation to Personal Loan")
continuous_col = ['Age', 'Income', 'ZIP_Code', 'CC_Avg','Mortgage']
for col in continuous_col:
    sns.distplot(df[col])
    plt.show()
sns.countplot(df['Personal_Loan'],label="Count")
y = df['Personal_Loan']
x = df.drop('Personal_Loan',axis=1)
x.columns
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:11]],axis=1)
data = pd.melt(data,id_vars="Personal_Loan",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="Personal_Loan", data=data,split=True, inner="quart")
plt.xticks(rotation=90)
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="Personal_Loan", data=data)
plt.xticks(rotation=90)
f, ax = plt.subplots(2, 2, figsize = (15,10))
plt.subplots_adjust(hspace=0.4)

sns.scatterplot(x = df['Mortgage'], y = df['Income'], hue = df['Personal_Loan'], alpha=0.4,ax = ax[0,0])
ax[0,0].set_title("Income vs Mortgage", size = 20, weight = "bold")
sns.scatterplot(x = df['ZIP_Code'],y=df['CC_Avg'], hue = df['Personal_Loan'], alpha=0.4, ax = ax[0,1])
ax[0,1].set_title("CC Avg vs Zip Code", size = 20, weight = "bold")
sns.scatterplot(x = df['Mortgage'], y = df['CC_Avg'], hue = df['Personal_Loan'], alpha=0.4,ax = ax[1,0])
ax[1,0].set_title("CC Avg vs Mortgage", size = 20, weight = "bold")
sns.scatterplot(x = df['Income'], y = df['CC_Avg'], hue = df['Personal_Loan'], alpha=0.4,ax = ax[1,1])
ax[1,1].set_title("CC Avg vs Income", size = 20, weight = "bold")
print("Income vs Mortgage vs CC Avg vs Personal_Loan")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X = df.drop(['Personal_Loan'], axis=1)
y = df[['Personal_Loan']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
!pip install lazypredict
from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(ignore_warnings=True)
model, predictions = clf.fit(X_train, X_test, y_train, y_test)
model
!pip install --upgrade scikit-learn==0.20.3
!pip install pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.tree import export_graphviz
final_model = DecisionTreeClassifier(criterion='entropy',max_depth=4, min_samples_split=2)
final_model.fit(X_train, y_train.ravel())
from sklearn.metrics import confusion_matrix
fig_dims = (8, 6)
fig, ax = plt.subplots(figsize=fig_dims)
ax.set_title("Decision Tree Classifier", size = 15, weight = "bold")
cm = confusion_matrix(y_test,final_model.predict(X_test))
sns.heatmap(cm,annot=True,fmt="d")
dot_data = StringIO() 
features = ['Age', 'Income', 'ZIP_Code', 'Family', 'CC_Avg', 'Education', 'Mortgage', 'Securities_Account', 'CD_Account', 'Online', 'Credit_Card']
classes = ['No', 'Yes']
export_graphviz(final_model, out_file=dot_data, feature_names=features, class_names=classes, filled=True, rounded=True, special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())