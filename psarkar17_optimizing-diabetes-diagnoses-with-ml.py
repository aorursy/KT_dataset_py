import pandas as pd



# Import country diabetes data

dataset = pd.read_excel('../input/country-summary/country_summary.xlsx')



# Remove unnecessary columns

dataset = dataset.drop(labels=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9','Unnamed: 11', 'Unnamed: 12','Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17','Unnamed: 19', 'Unnamed: 20','Unnamed: 22','Unnamed: 24', 'Unnamed: 25','Unnamed: 27'],axis=1)



# Remove values within parantheses

dataset['Number of adults 20–79 years with diabetes in 1,000s (95% confidence interval)'] = dataset['Number of adults 20–79 years with diabetes in 1,000s (95% confidence interval)'].str.replace(r"\(.*\)","")



dataset['Number of adults 20–79 years with undiagnosed diabetes in 1,000s (95% confidence interval)'] = dataset['Number of adults 20–79 years with undiagnosed diabetes in 1,000s (95% confidence interval)'].str.replace(r"\(.*\)","")



# Create new dataset comparing country/territory with select columns

dataset = dataset[['Country or territory', 'Number of adults 20–79 years with diabetes in 1,000s (95% confidence interval)', 'Number of adults 20–79 years with undiagnosed diabetes in 1,000s (95% confidence interval)']]



# Rename columns

diab_country = dataset.rename(columns={'Number of adults 20–79 years with diabetes in 1,000s (95% confidence interval)': '# 20-79 yr Old Adults w/ Diabetes (1000s)', 'Number of adults 20–79 years with undiagnosed diabetes in 1,000s (95% confidence interval)': '# 20-79 yr Old Adults w/ Undiagnosed Diabetes (1000s)'})



# Turn values from strings with commas and unecessary spaces into floating pt. numbers

diab_country['# 20-79 yr Old Adults w/ Diabetes (1000s)'] = diab_country['# 20-79 yr Old Adults w/ Diabetes (1000s)'].str.replace(',', '').str.replace(' ', '').astype(float)



diab_country['# 20-79 yr Old Adults w/ Undiagnosed Diabetes (1000s)'] = diab_country['# 20-79 yr Old Adults w/ Undiagnosed Diabetes (1000s)'].str.replace(',', '').str.replace(' ', '').astype(float)



diab_country
import pandas as pd



# Import GDP per capita data

gdp_capita = pd.read_csv('../input/gdp-per-capita/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1217696.csv')



# We only need 2019 data, so all other columns can be removed

# Additionally, all NaN terms and repeated data points are removed

# Right column is renamed



gdp_capita = (gdp_capita[['Country Name','2019']].rename(columns = {'2019':'GDP/Capita 2019'})).dropna().drop_duplicates()



gdp_capita
# Turn GDP/capita values into a dictionary, so GDP/capita can be easily fetched given a country/territory name

gdp_capita_np = gdp_capita.to_numpy() # turn to numpy array to extract data easily

gdp_capita_dict = {} # new empty dictionary to be filled



for n in range(len(gdp_capita_np)):

    gdp_capita_dict[gdp_capita_np[n][0]] = gdp_capita_np[n][1]

import numpy as np

# Turn each column into an individual numpy array

country_col = diab_country['Country or territory'].to_numpy()

col_1 = diab_country['# 20-79 yr Old Adults w/ Diabetes (1000s)'].to_numpy()

col_2 = diab_country['# 20-79 yr Old Adults w/ Undiagnosed Diabetes (1000s)'].to_numpy()



# Use GDP/Capita dictionary to turn country/territory names to GDP values

for i in range(len(country_col)):

    if country_col[i] in gdp_capita_dict:

        country_col[i] = gdp_capita_dict[country_col[i]]

        

# Not all countries/territories are present in GDP/Capita dictionary, so the remaining will be discarded

# GDPs will be appended to new array



discarded_countries = [] # Type: str

discarded_country_indices = [] # Array type: contains individual numpy arrays with indices

GDPs = [] # Array type: float



for i in country_col:

    if type(i) == str:

        discarded_countries.append(i)

        discarded_country_indices.append(np.where(country_col==i))

    else:

        GDPs.append(i)

        

GDPs = np.array(GDPs)

place_holder = []



for u in range(len(discarded_country_indices)):

    place_holder.append(discarded_country_indices[u][0][0])

discarded_country_indices = place_holder # Array type: int, contains indices of all countries without corresponding GDP/capita values



# The following code removes the diabetes data from countries without GDP/capita data from col_1 and col_2 (discarded_country_indices tells which indices to remove)



col_1 = np.delete(col_1,discarded_country_indices)

col_2 = np.delete(col_2,discarded_country_indices)



print("GDP Numpy Array Shape (Ind. variable): ",np.shape(GDPs))

print("Col 1 & 2 Shape (Dep. variables): ",np.shape(col_1),np.shape(col_2))



# Data is all in the correct form and shape and ready to be plotted.
import matplotlib.pyplot as plt



# Col_1 contains data on the # of 20-79 yr old adults w/ Diabetes

# Following plot compares Col_1 values to country GDP/capita



plt.scatter(GDPs,col_1)

plt.xlabel('GDP Per Capita')

plt.ylabel('# 20-79 yr Old Adults w/ Diabetes (1000s)')

plt.suptitle('Instances of Diabetes vs. Country GDP/Capita')



plt.show() 
# Visual inspection of the data indicates that there are two outlier countries with low GDP/Capita values and very high numbers of individuals with diabetes. For appropriate interpretation, these values will be excluded.



plt.scatter(GDPs,col_1)

plt.xlim(0,9e4)

plt.ylim(0,3e3)

plt.xlabel('GDP Per Capita')

plt.ylabel('# 20-79 yr Old Adults w/ Diabetes (1000s)')

plt.suptitle('Instances of Diabetes vs. Country GDP/Capita')

plt.show()



# General trend indicates that, with larger GDP per capita values, number of individuals with diabetes decreases
# Col_2 contains data on the # of 20-79 yr old adults that remain undiagnosed

# Following plot compares Col_2 values to country GDP/capita



plt.scatter(GDPs,col_2)

plt.xlabel('GDP Per Capita')

plt.ylabel('# 20-79 yr Old Adults w/ Undiagnosed Diabetes (1000s)')

plt.suptitle('Instances of Undiagnosed Diabetes vs. Country GDP/Capita')



plt.show()
# Similarly, the two points out of 155 countries in Col_2 will influence the analysis. For robust and appropriate interpretation, the two points will be removed from Col_2.



# A formal statistical test for the influence of those two points could be conducted in the future. However, that would be outside the scope of this analysis. A visual inspection was sufficient for now.



removed_indices = []

for value in col_2:

    if value > 4e4:

        removed_indices.append(np.where(col_2==value))

removed_indices = [removed_indices[0][0][0],removed_indices[1][0][0]]



GDPs = np.delete(GDPs,removed_indices)

col_2 = np.delete(col_2,removed_indices)



# Data ready to be plotted
from scipy import stats



plt.scatter(GDPs,col_2)

plt.xlabel('GDP Per Capita')

plt.ylabel('# 20-79 yr Old Adults w/ Undiagnosed Diabetes (1000s)')

plt.suptitle('Instances of Undiagnosed Diabetes vs. Country GDP/Capita')



# Add regression line:



m,b,r,p,std_err = stats.linregress(GDPs,col_2)

plt.plot(GDPs,GDPs*m+b)



# Calculate r-squared value:



r_squared = r**2



# Plot of Individuals with Undiagnosed Diabetes vs. GDP/Capita excluding two outlier points.



print("Slope: ",m,',',"Y-intercept: ",b)

print('R-squared: ',r_squared)

plt.show()
# Change window to (0,2000):

plt.scatter(GDPs,col_2)

plt.xlabel('GDP Per Capita')

plt.ylabel('# 20-79 yr Old Adults w/ Undiagnosed Diabetes (1000s)')

plt.suptitle('Instances of Undiagnosed Diabetes vs. Country GDP/Capita')



plt.plot(GDPs,GDPs*m+b)

plt.ylim(0,2e3)



print("Slope: ",m,',',"Y-intercept: ",b)

print('R-squared: ',r_squared)

print()

print("Decrease in number of individuals with undiagnosed diabetes with a $20,000 increase in GDP/Capita (1000s): ",abs(m*2e4))



plt.show()
# Split Col_2 values into Lower & Upper:



Lower,Upper = col_2,col_2



split = 4e3

indices_to_remove_for_lower = []

for value in GDPs:

    if value >= split:

        indices_to_remove_for_lower.append(np.where(GDPs==value))



indices_to_remove_for_upper = []

for value in GDPs:

    if value < split:

        indices_to_remove_for_upper.append(np.where(GDPs==value))



place_holder = []

for b in range(len(indices_to_remove_for_lower)):

    place_holder.append(indices_to_remove_for_lower[b][0][0])

indices_to_remove_for_lower = place_holder



place_holder = []

for b in range(len(indices_to_remove_for_upper)):

    place_holder.append(indices_to_remove_for_upper[b][0][0])

indices_to_remove_for_upper = place_holder



Lower = np.delete(Lower,indices_to_remove_for_lower)

Upper = np.delete(Upper,indices_to_remove_for_upper)



# Col_2 now split into the 2 categories above.



# Lower is the group containing # of individuals with undiagnosed diabetes from countries with LOW GDP/Capitas



# Upper is the group containing # of individuals with undiagnosed diabetes from countries with HIGH GDP/Capitas



print('n_lower: ',len(Lower),',','n_upper: ',len(Upper))

print()



#t-test

t_stat,pvalue = stats.ttest_ind(Lower,Upper,equal_var=False)

print('t-statistic: ',t_stat)

print('p-value: ',pvalue)
# Import data:



import pandas as pd

df_pima = pd.read_csv('../input/pimadiabetes/diabetes.csv')

df_pima.head()
# Split data

from sklearn.model_selection import train_test_split



X = df_pima.drop(labels='Outcome',axis=1).to_numpy()

y = df_pima['Outcome'].to_numpy()



X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)

y_test,y_train = y_test.reshape(-1),y_train.reshape(-1)



print("Train shape: ",np.shape(X_train),np.shape(y_train))

print("Test shape: ",np.shape(X_test),np.shape(y_test))



# We are now ready to proceed to the first algorithm: SVM
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



# Perform Hyperparamter Optimization:

# Optimization was run locally, so only code and output will be shown.



# Create a dictionary with all possible combinations of important hyperparamters for GridSearch to test

"""

params_to_test_svm = {'C':[0.01,0.1,1,10,100],'kernel':['linear','rbf','sigmoid'],'gamma':[0.001,0.01,0.1,1,10,100]}



grid = GridSearchCV(SVC(), params_to_test_svm, verbose=10, n_jobs=-1)

grid.fit(X_train,y_train)



print("best score: ", grid.best_score_)

print("best estimator: ", grid.best_estimator_)

print("best params: ", grid.best_params_)

"""

# Time elapsed: 35.1 minutes 



# best score:  0.768418308831196



# best estimator:  SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)



# best params:  {'C': 10, 'gamma': 0.001, 'kernel': 'linear'}



# Using the parameters found, performance is measured on test data:



model = SVC(C=10, gamma=0.001, kernel='linear')

model.fit(X_train,y_train)



# Find number of True Positives (tp), True Negatives (tn), False Positives (fp), and False Negatives (fn)



predicted = model.predict(X_test)

tp,tn,fp,fn=0,0,0,0



for value in range(len(predicted)):

    if predicted[value] == 0:

        if y_test[value] == predicted[value]:

            tn += 1

        else:

            fn += 1

    else:

        if y_test[value] == predicted[value]:

            tp += 1

        else:

            fp += 1

          

print("SVM Performance Summary:")

print()

print("# True Positives: ",tp)

print("# True Negatives: ",tn)

print("# False Positives: ",fp)

print("# False Negatives: ",fn)

print("Total: ",sum([tp,tn,fp,fn]))

print()



precision, recall, accuracy = tp/(tp+fp), tp/(tp+fn), (tp+tn)/(tp+tn+fn+fp)

f1 = 2*precision*recall/(precision+recall)

print("Precision Score: ",precision)

print("Recall Score: ",recall)

print("F1: ",f1)

print("Accuracy Score: ",accuracy)



# This array will be used to compare against against scores from other classifiers

svm_array = [precision,recall,f1,accuracy]
from sklearn.ensemble import RandomForestClassifier



"""

params_to_test_rf = {'n_estimators':[100,250,500,1000],'criterion':['gini','entropy'],'max_features':['auto','sqrt','log2']}

grid = GridSearchCV(RandomForestClassifier(),params_to_test_rf,n_jobs=-1)



print("best score: ", grid.best_score_)

print("best estimator: ", grid.best_estimator_)

print("best params: ", grid.best_params_)

"""

# Time elapsed: 1.2 minutes



# best score:  0.7703185392509663



# best estimator:  RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,criterion='entropy', max_depth=None, max_features='sqrt', max_leaf_nodes=None, max_samples=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False)



# best params:  {'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 500}



model = RandomForestClassifier(criterion='entropy',max_features='sqrt',n_estimators=500,n_jobs=-1)

model.fit(X_train,y_train)



predicted = model.predict(X_test)

tp,tn,fp,fn=0,0,0,0





for value in range(len(predicted)):

    if predicted[value] == 0:

        if y_test[value] == predicted[value]:

            tn += 1

        else:

            fn += 1

    else:

        if y_test[value] == predicted[value]:

            tp += 1

        else:

            fp += 1

          

print("RF Performance Summary:")

print()

print("# True Positives: ",tp)

print("# True Negatives: ",tn)

print("# False Positives: ",fp)

print("# False Negatives: ",fn)

print("Total: ",sum([tp,tn,fp,fn]))

print()



precision, recall, accuracy = tp/(tp+fp), tp/(tp+fn), (tp+tn)/(tp+tn+fn+fp)

f1 = 2*precision*recall/(precision+recall)

print("Precision Score: ",precision)

print("Recall Score: ",recall)

print("F1: ",f1)

print("Accuracy Score: ",accuracy)



rf_array = [precision,recall,f1,accuracy]
from sklearn.ensemble import GradientBoostingClassifier



"""

params_to_test_gbt = {'loss':['deviance'],'learning_rate':[0.01,0.05,0.1,0.2,0.3],'n_estimators':[100,300,500,1000],'criterion':['friedman_mse','mse','mae'],'min_samples_split':[2,5,10],'min_impurity_decrease':[0,0.01],'max_depth':[2,3,5,7]}



grid = GridSearchCV(GradientBoostingClassifier(),params_to_test_gbt,verbose=1,n_jobs=-1)

grid.fit(X_train,y_train)



print("best score: ", grid.best_score_)

print("best estimator: ", grid.best_estimator_)

print("best params: ", grid.best_params_)

"""



# best score: 0.7833799813407969



# best estimator: GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,learning_rate=0.1, loss='deviance', max_depth=3,max_features=None, max_leaf_nodes=None,min_impurity_decrease=0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=10,min_weight_fraction_leaf=0.0, n_estimators=100,n_iter_no_change=None, presort='deprecated',random_state=None, subsample=1.0, tol=0.0001,validation_fraction=0.1, verbose=0,warm_start=False)



# best params:  {'criterion': 'friedman_mse', 'learning_rate': 0.1, 'loss': 'deviance', 'max_depth': 3, 'min_impurity_decrease': 0, 'min_samples_split': 10, 'n_estimators': 100}



model = GradientBoostingClassifier(criterion='friedman_mse',learning_rate=0.1,loss='deviance',max_depth=3,min_impurity_decrease=0,min_samples_split=10,n_estimators=100)

model.fit(X_train,y_train)



predicted = model.predict(X_test)

tp,tn,fp,fn=0,0,0,0



for value in range(len(predicted)):

    if predicted[value] == 0:

        if y_test[value] == predicted[value]:

            tn += 1

        else:

            fn += 1

    else:

        if y_test[value] == predicted[value]:

            tp += 1

        else:

            fp += 1

            

print("GBT Performance Summary:")

print()

print("# True Positives: ",tp)

print("# True Negatives: ",tn)

print("# False Positives: ",fp)

print("# False Negatives: ",fn)

print("Total: ",sum([tp,tn,fp,fn]))

print()



precision, recall, accuracy = tp/(tp+fp), tp/(tp+fn), (tp+tn)/(tp+tn+fn+fp)

f1 = 2*precision*recall/(precision+recall)

print("Precision Score: ",precision)

print("Recall Score: ",recall)

print("F1: ",f1)

print("Accuracy Score: ",accuracy)



rf_array = [precision,recall,f1,accuracy]