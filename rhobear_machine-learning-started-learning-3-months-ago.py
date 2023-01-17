



import pandas as pd

import numpy as np



df = pd.read_csv('../input/data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})



dfsales = df.groupby('Description')['UnitPrice'].sum()

dfsales = pd.DataFrame(dfsales)

dfsales = dfsales.sort_values('UnitPrice', ascending = False)

dfsales['Description'] = dfsales.index



dfsales = dfsales[0:1000] # Memory issues after 500 products. UPDATED, KAGGLE IS MUCH FASTER



columns_to_keep = ['Description','CustomerID']

df1 = df[columns_to_keep]

uniqueproducts = dfsales['Description'].unique()



halfproducts = uniqueproducts[4:1000] # Memory issues after 500 products. UPDATED, KAGGLE IS MUCH FASTER

df1 = df1.where(df1['Description'].isin(halfproducts)).dropna()

df1 = df1.set_index('CustomerID')



df2 = pd.get_dummies(df1)

df2 = df2.reset_index()



uniqueproducts = df2.columns.values



listoflists = []



for item in uniqueproducts[1:]: # Groupby was very slow for entire frame. I iterated through, and it was faster

    df3 = df2.groupby('CustomerID')[item].max()

    listoflists.append(df3.values)

    customerlist = df3.index



    

newdf = pd.DataFrame(listoflists)

newdf['Product Name'] = uniqueproducts[1:]

newdf = newdf.set_index('Product Name')

newdf = newdf.T

newdf['CustomerID'] = customerlist

newdf = newdf.set_index('CustomerID')    







#loop through all columns to get test score for each product. USING RECALL as test score for model



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier





listofproducts = []

listofrecall = []



for column,name in enumerate(newdf.columns.values):

    columntotest = column

    excludedcolumn = newdf.columns.values[columntotest]

    X_sales = newdf.iloc[:,0:]

    X_sales = X_sales.drop(excludedcolumn, axis=1, level=None, inplace=False, errors='raise')

    y_sales = newdf.iloc[:,columntotest]

    X_train, X_test, y_train, y_test = train_test_split(X_sales, y_sales, random_state=0)   

    clf = DecisionTreeClassifier(max_depth = 10, random_state=0).fit(X_train, y_train)

    clf_predicted = clf.predict(X_test)

    confusion = confusion_matrix(y_test, clf_predicted)

    precision = precision_score(y_test, clf_predicted)

    recall = recall_score(y_test, clf_predicted)

    print (excludedcolumn)

    print ('Precision Score: {:.2f}'.format(precision))

    print ('Recall Score: {:.2f}'.format(recall))

    listofproducts.append(excludedcolumn)

    listofrecall.append(recall)

accuracydf = pd.DataFrame({'Recomendation': listofproducts, 'Score': listofrecall})



accuracydf.mean()





# loop through all customers and products *** GET PROBABILITIES ***



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier



mastervaluelist = []

masterproductlist = []

masterprobabilitylist = []



columnvalues = newdf.columns.values

columnvalues = columnvalues[0:-1]



for column,name in enumerate(columnvalues):

    print ('predicting product number: ', column, ':', name)



    columntotest = column

    excludedcolumn = newdf.columns.values[columntotest]

    X_sales = newdf.iloc[:,0:]

    X_sales = X_sales.drop(excludedcolumn, axis=1, level=None, inplace=False, errors='raise')

    y_sales = newdf.iloc[:,columntotest]

    X_train, X_test, y_train, y_test = train_test_split(X_sales, y_sales, random_state=0)

    clf = DecisionTreeClassifier(max_depth = 10, random_state=0).fit(X_train, y_train)

    lookforcustomers = X_test.where(newdf[excludedcolumn] == 0).dropna()





    clf_predicted = clf.predict(lookforcustomers)

    logs = clf.predict_proba(lookforcustomers)

    



    probabilitylist = []

    for item in logs:

        try:

            probabilitylist.append(float(item[1]))

        except:

            probabilitylist.append(float(0.0))

        

    predicteddf = pd.DataFrame({'Customer Tested': lookforcustomers.index,'Prediction': probabilitylist})

    predicteddf = predicteddf.set_index('Customer Tested')

    predictedvalues = predicteddf.index.values

    predictedvalues = predictedvalues.tolist()

    mastervaluelist = mastervaluelist + predictedvalues

    predictedproduct = [excludedcolumn] * len(predictedvalues)

    masterproductlist = masterproductlist + predictedproduct

    masterprobabilitylist = masterprobabilitylist + probabilitylist



    

    # Organize all customers, probabilities, products, and scores into a dataframe



finaldf = pd.DataFrame({'Customer': mastervaluelist, 'Recomendation' : masterproductlist, 'Probability': masterprobabilitylist})





finalfinaldf = pd.merge(finaldf,accuracydf, on = 'Recomendation', how = 'outer')

finalfinaldf.sort_values(['Customer','Score'], ascending = False)



finalfinaldf['WeightedScore'] = finalfinaldf['Probability'] * finalfinaldf['Score']



finalfinaldf = finalfinaldf.where(finalfinaldf['WeightedScore'] != 0).dropna().sort_values(['Customer', 'WeightedScore'], ascending = False)

finalfinaldf.groupby('Customer').head(3)