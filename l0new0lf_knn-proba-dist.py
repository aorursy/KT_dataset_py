X = [

        [0], 

        [1], 

        [2], 

        [3]

]

y = [0, 0, 1, 1]



from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X, y)



print('Simply predicting class: ', neigh.predict([[1.1]]))

#[0]



print('Predicting w/ probabilities: ', neigh.predict_proba([[0.9]]))

#[[0.66666667 0.33333333]]