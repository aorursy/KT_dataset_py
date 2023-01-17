print("Machine Learning for Interest Classification System based Decision Tree Method")
from sklearn import tree
#Data training untuk Machine Learning



a = [

        [4,2,1,2,2],

        [1,1,4,1,1],

        [3,1,2,2,3],

        [4,1,4,1,5],

        [4,3,3,1,6],

        [4,1,3,1,1],

        [4,3,3,1,7],

        [4,1,1,1,3],

        [4,1,2,1,6],

        [3,1,1,1,6],

        [2,1,1,1,7],

        [3,1,1,1,3],

        [2,1,1,1,6],

        [3,1,1,2,4],

        [3,2,1,2,7],

        [4,1,1,2,2],

        [2,2,1,2,3],

        [4,2,1,1,7],

        [3,3,1,1,3],

        [1,1,2,1,3]

    ]





b = [

     'politik',

     'sains dan budaya',

     'sains',

     'sains',

     'Budaya',

     'sains',

     'Budaya',

     'politik dan budaya',

     'sains',

     'budaya',

     'politik',

     'sains dan budaya',

     'sains',

     'politik',

     'budaya',

     'politik',

     'sains',

     'politik',

     'budaya',

     'sains' 

    ]
clf = tree.DecisionTreeClassifier()



clf = clf.fit(a,b)
prediksi = clf.predict([[2,1,4,1,2]])

prediksi