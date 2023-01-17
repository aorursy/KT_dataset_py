import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df1 = pd.read_csv(r"../input/black-friday/BlackFriday.csv")
df1.head()
df1 = df1.fillna(0)
new_df = df1.Stay_In_Current_City_Years.replace(['4+'],4)
new_df.head(10)
df1['new_stay'] = new_df
df1.head(10)
new_df = df1.Age.replace(['0-17','18-25','26-35','36-45','46-50','51-55','55+'],[0,1,2,3,4,5,6])
new_df.head(10)
df1['new_age'] = new_df
df1.head(10)
new_df = df1.City_Category.replace(['A','B','C'],[0,1,2])
new_df.head(10)
df1['new_city'] = new_df
df1.head(10)
new_df = df1.Gender.replace(['M','F'],[0,1])
new_df.head(10)
df1['new_gender'] = new_df
df1.head(10)
AgeGroup = df1['Age'].map(lambda n: n.split("|")[0].split(":")[0]).value_counts().head(20)
AgeGroup.plot.bar()
sns.countplot(df1['Gender'])
sns.countplot(df1['City_Category'])
sns.countplot(df1['Age'],hue=df1['Gender'])
sns.countplot(df1['Occupation'],hue=df1['Gender'])
sns.countplot(df1['Purchase'],hue=df1['City_Category'])
print(df1['Gender'].value_counts())
df1_Male = df1.loc[df1['Gender'] == 'M']
df1_Female = df1.loc[df1['Gender'] == 'F']
total_spending_male = df1_Male['Purchase'].sum()
total_spending_male
total_spending_female = df1_Female['Purchase'].sum()
total_spending_female
spending_data = [['M',total_spending_male],['F',total_spending_female]]
df2 = pd.DataFrame(spending_data, columns=('Gender','Purchase'))
df2
df1_A = df1.loc[df1['City_Category'] == 'A']
df1_B = df1.loc[df1['City_Category'] == 'B']
df1_C = df1.loc[df1['City_Category'] == 'C']
total_spending_A = df1_A['Purchase'].sum()
total_spending_A
total_spending_B = df1_B['Purchase'].sum()
total_spending_B
total_spending_C = df1_C['Purchase'].sum()
total_spending_C
### Total Spent by each city category.
spending_data_City = [['A',total_spending_A],['B',total_spending_B],['C',total_spending_C]]
df3 = pd.DataFrame(spending_data_City, columns=('City_Category','Purchase'))
df3
df1['combined_G_M'] = df1.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)
print(df1['combined_G_M'].unique())
df5 = df1.groupby(['combined_G_M','Age']).size()
df5
sns.countplot(df1['Age'],hue=df1['combined_G_M'])
sns.boxplot('Age','Purchase', data = df1)
plt.show()
Age_buy = df1.groupby(["Age"])["Purchase"].sum()
Age_buy.plot.bar()
Occu_buy = df1.groupby(["Occupation"])["Purchase"].sum()
Occu_buy.plot.bar()
City_buy = df1.groupby(["City_Category"])["Purchase"].sum()
City_buy.plot.bar()
product_age = df1.groupby(["Age"])["Product_Category_1", "Product_Category_2", "Product_Category_3"].sum()
product_age.plot.bar()

df1.Purchase.describe()
df10 = df1.head(1000)
df10.plot.scatter(x = "User_ID",y="Purchase")
df1.corr()
clean_data = df1.copy()
clean_data['good_customer'] = (clean_data['Purchase'] > 12073)*1
print(clean_data['good_customer'])
y=clean_data[['good_customer']].copy()
y.head()
df1.columns
customer_features = ['new_age', 'Occupation', 'new_city',
       'new_stay', 'Marital_Status','new_gender', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3']
X = clean_data[customer_features].copy()
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
good_customer_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
good_customer_classifier.fit(X_train, y_train)
predictions = good_customer_classifier.predict(X_test)
predictions[:10]
y_test['good_customer'][:10]
accuracy = accuracy_score(y_true = y_test, y_pred = predictions)
print(accuracy * 100)
