from sklearn import metrics

from sklearn.datasets import load_boston

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split 





from itertools import combinations 



import pandas as pd



import matplotlib.pyplot as plt

# We will not go far, download the built-in dataset in the library to solve the regression problem

boston = load_boston()



data = pd.DataFrame(boston.data, columns = boston.feature_names)

y = boston.target
def combination(data, y, column_names):

    """

        this function is not applicable in real life, as it goes through many combinations and

        is too complicated. But there are not many lines in this dataset, so you can do this. 

    """

    metrics_table = pd.DataFrame(columns = ["RSS", "r^2", "params", "coef"])

    

    for i in range(1,len(column_names)):



        comb = combinations(column_names, i) 

        

        for e in list(comb): 



            X = data[list(e)]

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

            

            regr = LinearRegression()

            regr.fit(X_train, y_train)

            y_pred = regr.predict(X_test)



            metrics_table = metrics_table.append({"RSS":((y_pred - y_test) ** 2).sum(),\

                                                  "r^2": metrics.r2_score(y_test, y_pred),\

                                                  "params":e,\

                                                  "coef": regr.coef_}, ignore_index=True)





    return metrics_table 

    

metrics_table = combination(data, y, boston.feature_names)            
plt.figure(figsize=(10,8))

plt.scatter(metrics_table.iloc[:, 0], metrics_table.iloc[:, 1], color='green')

plt.xlabel('Residual sum of squares')

plt.ylabel('Coefficient of determination')

plt.title('Dependency visualization')

plt.show()
metrics_table.sort_values(by=['RSS']).head(2)
params_1 = metrics_table.iloc[6919, :]['params']

params_2 = metrics_table.iloc[7955, :]['params']



print("Restricted model number 2 has no {0} parameters".format(set(params_1) ^ set(params_2)))