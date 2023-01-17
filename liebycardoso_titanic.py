#importando as bibliotecas
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb
import pandas as pd
import numpy as np
#Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import  model_selection, tree, preprocessing
titanic_df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

titanic_df.append(test)
titanic_df.head()
# Verifica  o tipo dos dados criados
titanic_df.dtypes
((len(titanic_df) - titanic_df.count()) / len(titanic_df)) *100
titanic_df.Age = titanic_df.Age.fillna(titanic_df.Age.mean())
# Create a new column: Family
titanic_df['Family'] = titanic_df['SibSp'] + titanic_df['Parch']
def age_range(idade):
    """ 
    Returns the age range for the given age.

     Args:
         Age: Value number representing age.
     Returns:
         Returns a string with the age range identified for age.
         Values Domain = Elderly, Adult, Young Adult, Teen and Child.
    """    
    
    if idade >= 65:
        return 'Elderly'
    elif idade >= 33:
        return 'Adult'
    elif idade >= 18:
        return 'Young Adult'
    elif idade >= 12:
        return 'Teen'
    else:
        return 'Child'    
# Calls the age_range function passing as the Age column parameter and assigns the result to the new AgeRange column
titanic_df['AgeRange']= titanic_df.Age.apply(age_range) 
# Print chart with total passengers by age group.
tempAgeRange = titanic_df.groupby(['AgeRange']).size()
sb.barplot(tempAgeRange.index, tempAgeRange);
# Delete columns
titanic_df.drop(['SibSp', 'Parch', 'Cabin','Embarked', 'Ticket','Name'], axis=1, inplace=True)

titanic_df.head()
titanic_df.describe()
((len(titanic_df.Family) - titanic_df[titanic_df['Family']>0].count()) / len(titanic_df.Family)) *100
# List the unique values of family
titanic_df.Family.unique()
acompanhante_sobrevivencia = titanic_df.groupby(['Family','Survived']).size()
sb.heatmap(acompanhante_sobrevivencia.unstack(), annot=True, fmt='g')
plt.xlabel('0 - Mortos , 1 - Sobreviventes')
plt.ylabel('Total Acompanhantes')
# Sort the data in descending order to see if there are many high values such as max = 512.
titanic_df.Fare.sort_values(ascending=False).head()
# Lists the 10 largest Fare values, but the list has repeated values, then filters
# Only unique (.unique) and returns the second line [1], because the first is the Max. 
second_max_fare = titanic_df.Fare.nlargest(10).unique()[1]
# Assigns the second highest value in the column (second_max_fare)
# Where Fare is equal to the maximum value of the column
titanic_df.Fare = titanic_df.Fare.apply(lambda x: second_max_fare if x==titanic_df.Fare.max() else x)
# Checks whether the max has been changed to the second value.
titanic_df.Fare.max()
# Search result where the value of the passage is equal to 0.
titanic_df[titanic_df['Fare']==0]
# Returns the average of the Fare column when the value passed as parameter is equal to 0.
titanic_df.Fare = titanic_df.Fare.apply(lambda x: titanic_df.Fare.mean() if x==0 else x)  
# Summary
titanic_df.Fare.describe()
# Assign an integer value to the categorized values of Sex (0 - female, 1 - male)
titanic_df['SexInt'] = map(int, titanic_df.Sex == 'male')
# Calls the correlation method by passing the pearson type as a parameter and saves the values in a dataframe.
correlation_df = titanic_df.corr(method='pearson', min_periods=1).abs()
correlation_df
correlation_df.unstack().sort_values(ascending=False)
# Overview of all variables with the help of pairplot ().
sb.pairplot(titanic_df, hue='Survived',  size=2.5, markers=['o','s'], palette=['gray','red']);
fig = plt.figure(figsize=(18,6), dpi=1600) 

ax1 = plt.subplot(2,2,1)
# Histogram Column Age
titanic_df.Age.hist(bins=10) 
plt.xlabel("Age")
plt.title("Histrogram Age, (bin=10)")    

# Create subplot2 
ax2 = plt.subplot(2,2,2)
# Plot density of column Age
titanic_df['Age'].plot(kind='kde', style='k--')
plt.ylabel("Density")
plt.xlabel("Age")
plt.title("Densidade - Age")

# Create subplot3
ax3 = plt.subplot(2,2,(3,4))
# Plot density - Class
titanic_df.groupby('Pclass').Age.plot.kde()
plt.xlabel("Age")  
plt.title("Distribution Age/Class")
plt.legend(('1 Class', '2 Class','3 Class'),loc='best') 
fig = plt.figure(figsize=(18,6), dpi=1600) 

ax1 = plt.subplot(1,2,1)
titanic_df.groupby('Pclass').Survived.plot.kde()

plt.xlabel("0 - Died  1 - Survived")
plt.ylabel("Density")
plt.title("Distribution Survived by Class")
plt.legend(('1 Class', '2 Class','3 Class'),loc='best') 

ax2 = plt.subplot(1,2,2)
titanic_df.groupby('Sex').count()['Survived'].plot.bar()
plt.xlabel("Female - Male")
plt.title("Number of female and male");
# Create survived
sobreviventes = titanic_df[titanic_df['Survived']==1]

sb.factorplot(x="Sex", y="Age", hue="Pclass",
               col="Pclass", data=sobreviventes, kind="box", size=4, aspect=.5)

# Label x and y
plt.xlabel("Genero")
plt.ylabel('Idade');
# Plot violinplot with survivor distribution by sex and age group
ax = sb.violinplot(data=titanic_df, x='SexInt', y='Survived', hue='AgeRange')
ax.set(xlabel='(0)Mulheres , (1)Homens', ylabel='(0)Mortos, (1)Sobreviventes')
# Group by Sex
titanic_df.groupby(['Sex']).mean()
# Group by Age Range and Sex 
titanic_df.groupby(['AgeRange','Sex']).mean()
# Mean of group by Pclass
titanic_df.groupby(['Pclass']).mean()
# Group data by age group, gender, class
faixa_etaria_genero = titanic_df.groupby(['AgeRange','Sex','Pclass']).mean()
faixa_etaria_genero
jovem_adulta = titanic_df.groupby(['AgeRange','Sex','Pclass']).mean().T
jovem_adulta
titanic_df.head()
# Create a copy of titanic_df
processed_df = titanic_df.copy()
# Delete SexInt column
processed_df.drop('SexInt',axis=1, inplace=True)
processed_df.head()
le = preprocessing.LabelEncoder()

# Sex and AgeRange columns receive their numerical version
processed_df.Sex = le.fit_transform(processed_df.Sex)
processed_df.AgeRange = le.fit_transform(processed_df.AgeRange)
# X Receives all values from the dataset minus the Survived column that will be used in the comparison
X = processed_df.drop(['Survived'], axis=1).values

# y Receives the values from the Survived column that will be used by the model as a comparison
y = processed_df['Survived'].values

# Divide the matrices into test and training
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
X_test[:,:1]
passengerID = X_test[:,:1]
X_train = X_train[:,1:]
X_test = X_test[:,1:]
# Create Decision Tree
clf_dt = tree.DecisionTreeClassifier(max_depth=5)

# Train model
clf_dt.fit(X_train, y_train)
# Check the accuracy of the model
clf_dt.score (X_test, y_test)
# Generates the GraphViz representation of the decision tree. The data is recorded in the file titanic_tree.dot
#Data can be viewed graphically at http://www.webgraphviz.com/
#tree.export_graphviz(clf_dt, out_file='titanic_tree.dot', feature_names=processed_df.columns[1:])
# Create Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, oob_score=True)

# Train model the same way we did with DecisionTreeClassifier 
clf_rf.fit(X_train, y_train)
# Predict result
Y_pred = clf_rf.predict(X_test)
# Check accuracy
clf_rf.score(X_train, y_train)
# Out-of-bag estimate (oob) error: 81%
clf_rf.oob_score_
passengerID.ravel()
passID = passengerID.T.flat
submission = pd.DataFrame({
        "PassengerId": passengerID.ravel(),
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)