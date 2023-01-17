import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv("/kaggle/input/insurance/insurance.csv")

data.head()
data.isnull().sum()
for col in data.columns:

    print(data[col].value_counts())
smoker={

    'yes':1,

    'no':0

}



region={

    'southwest':1,

    'southeast':2,

    'northeast':3,

    'northwest':4

}

gender={

    'male':1,

    'female':2

}



data['smoker']=data['smoker'].map(smoker)

data['region']=data['region'].map(region)

data['sex']=data['sex'].map(gender)
colors=["#31DFA2","#58D68D","#49c99e","#45B39D","#138D75","#167856"]
spearman_corr=data.corr(method='spearman') # spearman for finding non linear dependencies

spearman_heatmap = sns.heatmap(spearman_corr)

plt.sca(spearman_heatmap)

plt.title("Spearman Non-Linear Correlation")
# finding the distribution graph forage 

sns.distplot(data['age'], color = colors[0])
# finding unique values in age column 

age_slot=data['age'].unique()



# using two dictionaries one for smoker and anotther for non-smoker

smoker={}

nonsmoker={}



# collecting data about charges for each age value for both smoker and non-smoker

for slot in age_slot :

    

    # for smoker age vs charge with min , max and mean

    s_min_charge = data[(data['age']==slot) & (data['smoker']==1)]["charges"].min()

    s_max_charge = data[(data['age']==slot) & (data['smoker']==1)]["charges"].max()

    s_mean_charge = data[(data['age']==slot) & (data['smoker']==1)]["charges"].mean()



    # for non-smoker age vs charge with min , max and mean

    ns_min_charge = data[(data['age']==slot) & (data['smoker']==0)]["charges"].min()

    ns_max_charge = data[(data['age']==slot) & (data['smoker']==0)]["charges"].max()

    ns_mean_charge = data[(data['age']==slot) & (data['smoker']==0)]["charges"].mean()

    

    smoker[slot]=[s_min_charge,s_max_charge,s_mean_charge]

    nonsmoker[slot]=[ns_min_charge,ns_max_charge,ns_mean_charge]

    

# plotting graphs for both , total 6 graphs are plotted

fig, axs = plt.subplots(3, 2)

fig.set_size_inches(20, 15)



axs[0][0].bar(smoker.keys(),[ls[0] for ls in smoker.values()],color=colors[0])

axs[0][0].set_ylabel('Charges', fontsize=14)

axs[0][0].set_title('Smoker min Charge', fontsize=16)



axs[0][1].bar(smoker.keys(),[ls[0] for ls in nonsmoker.values()],color=colors[0])

axs[0][1].set_ylabel('Charges', fontsize=14)

axs[0][1].set_title('Non-Smoker min Charge', fontsize=16)



axs[1][0].bar(smoker.keys(),[ls[1] for ls in smoker.values()],color=colors[2])

axs[1][0].set_ylabel('Charges', fontsize=14)

axs[1][0].set_title('Smoker max Charge', fontsize=16)



axs[1][1].bar(smoker.keys(),[ls[1] for ls in nonsmoker.values()],color=colors[2])

axs[1][1].set_ylabel('Charges', fontsize=14)

axs[1][1].set_title('Non-Smoker max Charge', fontsize=16)



axs[2][0].bar(smoker.keys(),[ls[2] for ls in smoker.values()],color=colors[3])

axs[2][0].set_xlabel('Age', fontsize=14)

axs[2][0].set_ylabel('Charges', fontsize=14)

axs[2][0].set_title('Non-Smoker mean Charge', fontsize=16)



axs[2][1].bar(smoker.keys(),[ls[2] for ls in nonsmoker.values()],color=colors[3])

axs[2][1].set_xlabel('Age', fontsize=14)

axs[2][1].set_ylabel('Charges', fontsize=14)

axs[2][1].set_title('Non-Smoker mean Charge', fontsize=16)



fig.suptitle('Smoker vs Non-Smoker', fontsize=20)

plt.show()

underweight = data[(data['bmi']<=18.5)]["charges"]

healthy = data[(data['bmi']>=18.5) & (data['bmi']<=24.9)]["charges"]

overweight = data[(data['bmi']>=25) & (data['bmi']<=29.9)]["charges"]

obese = data[(data['bmi']>=30)]["charges"]



fig, axs = plt.subplots(2, 2)

fig.set_size_inches(20, 15)



sns.distplot(underweight , ax=axs[0][0] , color=colors[0])

axs[0][0].set_title('Underweight', fontsize=16)



sns.distplot(healthy , ax=axs[0][1] , color=colors[1])

axs[0][1].set_title('Healthy', fontsize=16)



sns.distplot(overweight , ax=axs[1][0] , color=colors[2])

axs[1][0].set_title('Overweight', fontsize=16)



sns.distplot(obese , ax=axs[1][1] , color=colors[4])

axs[1][1].set_title('Obese', fontsize=16)



fig.suptitle('BMI vs Charges', fontsize=20)

plt.show()
# children = data['children'].unique() # 6 unique values



fig, axs = plt.subplots(3, 2)

fig.set_size_inches(20, 15)



num_child=0



for i in range(3):

    

    for j in range(2):



        sns.distplot(data[ (data['children'] == num_child) ]["charges"], ax = axs[i][j] , color = colors[num_child] )

        axs[i][j].set_title('Number of children : '+ str(num_child), fontsize=16)

        num_child+= 1



fig.suptitle('Number of Children vs Charges', fontsize=20)

plt.show()
data[ (data['children'] == 5) ]
#region = data['region'].unique() # 4 unique values



region_string = ['Southwest', 'Southeast', 'Northeast', 'Northwest']



fig, axs = plt.subplots(2, 2)

fig.set_size_inches(20, 15)



region=1



for i in range(2):

    

    for j in range(2):



        sns.distplot(data[ (data['region'] == region) ]["charges"], ax = axs[i][j] , color = colors[region] )

        axs[i][j].set_title('Region : '+ str(region_string[region-1]), fontsize=16)

        region+= 1



fig.suptitle('Region vs Charges', fontsize=20)

plt.show()
# importing all the necessary libraries



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures



from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor



from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_classification
X = data.drop(['charges'], axis = 1) # independent attributes

Y = data.charges # target or dependent attribute



x_train, x_test, y_train, y_test = train_test_split(X,Y)
# linear model

linear=LinearRegression()

linear.fit(x_train, y_train)



# decision tree regressor

decision_tree = DecisionTreeRegressor(random_state=0)

decision_tree.fit(x_train, y_train)



# random forest regressor

random_forest = RandomForestRegressor(max_depth=2, random_state=0)

random_forest.fit(x_train, y_train)



# printing all the prediction results

print("Linear Regressor Score : ", linear.score(x_test,y_test))

print("Decision Tree Regressor Score : ", decision_tree.score(x_test,y_test))

print("Random Forest Regressor : ", random_forest.score(x_test,y_test))

# polynomial features for regression

quad = PolynomialFeatures (degree = 2)

x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)

poly_linear = LinearRegression().fit(X_train,Y_train)



print("Polynomial Linear Regressor : ", poly_linear.score(X_test,Y_test))