from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, naive_bayes
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
features = data['data']
labels = data['target']
labels_names = data['target_names']
features_names = data['feature_names']
print("Features: ", features_names)
print("Target: ", labels_names)
train, test, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.3, random_state=78)
# classifiers
svc = svm.SVC()
nb = naive_bayes.GaussianNB()
dtree = tree.DecisionTreeClassifier()
# training
svc_model = svc.fit(train, train_labels)
nb_model = nb.fit(train, train_labels)
dtree_model = dtree.fit(train, train_labels)
# test
predictions_svc = svc_model.predict(test)
predictions_nb = nb_model.predict(test)
predictions_dtree = dtree_model.predict(test)
# evaluation
print("Accuracy for SVM: {}".format(accuracy_score(test_labels, predictions_svc)))
print("Accuracy for Naive Bayes: {}".format(accuracy_score(test_labels, predictions_nb)))
print("Accuracy for Decision Tree: {}".format(accuracy_score(test_labels, predictions_dtree)))
