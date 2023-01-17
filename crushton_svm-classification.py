## =============== Part 1: Loading and Visualizing Data ================

#  



import pandas as pd



df = pd.read_csv('../input/Iris.csv')

# df.head(5)

# df.size()

y = df['Species'] # response vector

X_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

X = df[X_columns] 



import matplotlib.pyplot as plt



# Univariate Histograms                                                    

df.hist()

plt.show()



# Univariate Density Plots

df.plot(kind='density', subplots=True, layout=(3,2), sharex=False)      

plt.show()           
#======== MODEL ================

# Seeing that PetalLength and PetalWidth are bimodal

# It is likely they are the dominant features for our model



from sklearn import svm



# Create SVM classification object                                         

model = svm.svc(kernel='linear', c=1, gamma=1)                             

model.fit(X, y)

model.score(X, y)

                                                                           

#Predict Output

predicted= model.predict(X_test)   