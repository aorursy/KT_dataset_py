!pip install torchRDS
from torchRDS.RDS import RDS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
class LR:
    def run(self, state, action):
        data_x, data_y = state
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]
        
        clf = LogisticRegression(solver='liblinear')
        clf.fit(train_x, train_y)
        return clf.predict_proba(test_x)
    
class KNN:
    def run(self, state, action):   
        data_x, data_y = state
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_x, train_y)

        return clf.predict_proba(test_x)

class RF:
    def run(self, state, action):   
        data_x, data_y = state
        train_x, train_y, test_x = data_x[action == 1], data_y[action == 1, 1], data_x[action == 0]

        clf = RandomForestClassifier(n_estimators=16, bootstrap=False, n_jobs=-1)
        clf.fit(train_x, train_y)

        return clf.predict_proba(test_x)
trainer = RDS(data_file="../input/dccc-dataset/default_of_credit_card_clients.csv", target=[0], task="classification", measure="auc", 
              models=[LR(), KNN(), RF()], learn="deterministic", ratio=0.6, delta=0.02, weight_iid=0.1, iters=100, device="cuda")
sample = trainer.train()
print("Number of observations in the trainning set:", sum(sample))
print("Number of observations in the test set:", len(sample) - sum(sample))