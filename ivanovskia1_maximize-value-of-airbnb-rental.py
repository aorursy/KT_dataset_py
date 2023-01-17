## Read Data
import csv
import pandas as pd
import re
df = pd.read_csv("../input/New York.csv")
df.head()
df.describe()
df["RevPAR"] = (df['price'] * (30 - df['availability_30']))



df['amenities'][0]
sc_sub = re.compile('\W+')
df['amenities'] = [sc_sub.sub(' ', amenity) for amenity in df['amenities']]
print(df['amenities'][0])
amenities2 = ['Wireless Internet', 'Air conditioning', 'Pool', 'Kitchen',
       'Free parking on premises', 'Gym', 'Hot tub', 'Indoor fireplace',
       'Heating', 'Family kid friendly', 'Suitable for events', 'Washer',
       'Dryer', 'Essentials', 'Shampoo', 'Lock on bedroom door', 'Cable TV',
       '24 hour check in', 'Laptop friendly workspace', 'Hair dryer']
for amenity in amenities2:
    df[amenity] = df.amenities.str.contains(amenity)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

# Show the results of a linear regression within each dataset
sns.lmplot(x="availability_30", y="price", data=df)
plt.show()

# Show the results of a linear regression within each dataset
sns.lmplot(x="availability_30", y="RevPAR", data=df)
plt.show()
import numpy as np
df = df[np.abs(df.RevPAR-df.RevPAR.mean())<=(3*df.RevPAR.std())]
# Show the results of a linear regression within each dataset
sns.lmplot(x="availability_30", y="price", data=df)
plt.show()


# Show the results of a linear regression within each dataset
sns.lmplot(x="availability_30", y="RevPAR", data=df)
plt.show()
print("Average Monthly Revenue:", df["RevPAR"].mean())
print("Median Monthly Revenue:", df["RevPAR"].median())
# Rooms that make more than the average monthly revenue will be labeled as 1 (Successful),
# whereas rooms that earn less will be labeled as 0 (Not Successful)
AverageRevPAR = df["RevPAR"].mean()
df["Successful"] = [1 if x >= AverageRevPAR else 0 for x in df["RevPAR"]]
df.isnull().sum()
Exploratory_Analysis = ['Wireless Internet', 'Air conditioning', 'Pool', 'Kitchen',
       'Free parking on premises', 'Gym', 'Hot tub', 'Indoor fireplace',
       'Heating', 'Family kid friendly', 'Suitable for events', 'Washer',
       'Dryer', 'Essentials', 'Shampoo', 'Lock on bedroom door',
       '24 hour check in', 'Laptop friendly workspace', 'Hair dryer','is_business_travel_ready']
for r in Exploratory_Analysis :
    print(df.groupby(r)['Successful'].mean())
df['number_of_reviews'].fillna(0, inplace=True)
df['review_scores_rating'].fillna(0, inplace=True)
df['reviews_per_month'].fillna(0, inplace=True)
df.drop(["amenities","id","price","weekly_price", 
         "availability_30", "review_scores_accuracy", "review_scores_cleanliness","review_scores_checkin", 
         "review_scores_communication", "review_scores_location", "review_scores_value",
         "cleaning_fee", "security_deposit",
         "host_has_profile_pic",'bathrooms','bedrooms','beds', 'latitude','longitude',
         'accommodates','square_feet','guests_included','minimum_nights','maximum_nights'], axis=1, inplace = True)
X = df[[
        'neighbourhood_cleansed','is_location_exact',
       'bed_type',
        'number_of_reviews',
       'instant_bookable', 'cancellation_policy', 'Wireless Internet',
       'Air conditioning', 'Pool', 'Kitchen', 'Free parking on premises',
       'Gym', 'Hot tub', 'Indoor fireplace', 'Heating', 'Family kid friendly',
       'Suitable for events', 'Washer', 'Dryer', 'Essentials', 'Shampoo', 'Cable TV',
       'Lock on bedroom door', '24 hour check in', 'Laptop friendly workspace',
       'Hair dryer','is_business_travel_ready','number_of_reviews','review_scores_rating','reviews_per_month']]

y = df["Successful"]
X.isnull().sum()
df[df.columns[1:]].corr()['Successful'][:-1]
X_train1 = pd.get_dummies(X)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# Create Training and Test Dataset with 75% Training and 25% Test
X_train, X_test, y_train, y_test = train_test_split(X_train1, y, test_size=0.25)

# Run Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Analyze results
print("Results:")
print("Accuracy", metrics.accuracy_score(y_test,y_pred))

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

#Specificity: When the actual value is negative, how often is the prediction correct?
print("Specificity:",TN / float(TN + FP))

#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
print("False Positive Rate:",FP / float(TN + FP))

#Precision: When a positive value is predicted, how often is the prediction correct?
print("Precision:",metrics.precision_score(y_test, y_pred))

#Sensitivity:
print("Recall:",metrics.recall_score(y_test, y_pred))

print("-----------------------------------------------------------------------")
# examine the class distribution of the testing set (using a Pandas Series method)
print("Class Distribution:", y_test.value_counts())
# calculate the percentage of ones
print("Percentage of Ones:", y_test.mean())

# calculate the percentage of zeros
print("Percentage of Zeros:", 1 - y_test.mean())

# calculate null accuracy (for binary classification problems coded as 0/1)
print("Null Accuracy:",max(y_test.mean(), 1 - y_test.mean()))

print('------------')
print("Improvement in accuracy compared to Naive Model", metrics.accuracy_score(y_test,y_pred) - max(y_test.mean(), 1 - y_test.mean()))

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train1, y, test_size=0.25, random_state = 42)

tree = DecisionTreeClassifier(max_depth=8, random_state=0)
tree.fit(X_train2, y_train2)
y_pred2 = tree.predict(X_test2)
print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train2, y_train2)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test2, y_test2)))


# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test2, y_pred2)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

#Specificity: When the actual value is negative, how often is the prediction correct?
print("Specificity:",TN / float(TN + FP))

#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
print("False Positive Rate:",FP / float(TN + FP))

#Precision: When a positive value is predicted, how often is the prediction correct?
print("Precision:",metrics.precision_score(y_test2, y_pred2))

#Sensitivity:
print("Recall:",metrics.recall_score(y_test2, y_pred2))

print("--------------------------------------------------------------")
# examine the class distribution of the testing set (using a Pandas Series method)
print("Class Distribution:", y_test2.value_counts())
# calculate the percentage of ones
print("Percentage of Ones:", y_test2.mean())

# calculate the percentage of zeros
print("Percentage of Zeros:", 1 - y_test2.mean())

# calculate null accuracy (for binary classification problems coded as 0/1)
print("Null Accuracy:",max(y_test2.mean(), 1 - y_test2.mean()))

print("--------------")
print("Improvement in accuracy compared to Naive Model", metrics.accuracy_score(y_test2,y_pred2) - max(y_test2.mean(), 1 - y_test2.mean()))

a = zip(X_train1,tree.feature_importances_)
Important_Features = pd.DataFrame(list(a), columns = ['features','FeatureImportances'])
Important_Features.sort_values(by=['FeatureImportances'],ascending = False)
Avg_rev_ff = df.groupby('Family kid friendly')['RevPAR'].mean()
Suc_ff = df.groupby('Family kid friendly')['Successful'].mean()

print(Avg_rev_ff)
print(Suc_ff)
Avg_rev_Lock = df.groupby('Lock on bedroom door')['RevPAR'].mean()
Suc_Lock = df.groupby('Lock on bedroom door')['Successful'].mean()

print(Avg_rev_Lock)
print(Suc_Lock)
Avg_rev_AC = df.groupby('Air conditioning')['RevPAR'].mean()
Suc_AC = df.groupby('Air conditioning')['Successful'].mean()

print(Avg_rev_AC)
print(Suc_AC)
Avg_rev_BT = df.groupby('is_business_travel_ready')['RevPAR'].mean()
Suc_BT = df.groupby('is_business_travel_ready')['Successful'].mean()

print(Avg_rev_BT)
print(Suc_BT)