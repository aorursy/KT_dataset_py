import numpy as np


class Perceptron:
    def __init__(self, max_iters=100):
        self.max_iters = max_iters
    
    def fit(self, X, y):
        # Bookkeeping.
        X, y = np.asarray(X), np.asarray(y)
        iters = 0
        
        # Insert a bias column.
        X = np.concatenate((X, np.asarray([[1] * X.shape[0]]).T), axis=1)
        
        # Initialize random weights.
        ω = np.random.random(X.shape[1])        
        
        # Train as many rounds as allotted, or until fully converged.
        for _ in range(self.max_iters):
            y_pred_all = []
            for idx in range(X.shape[0]):
                x_sample, y_sample = X[idx], y[idx]
                y_pred = int(np.sum(ω * x_sample) >= 0.5)
                if y_pred == y_sample:
                    pass
                elif y_pred == 0 and y_sample == 1:
                    ω = ω + x_sample
                elif y_pred == 1 and y_sample == 0:
                    ω = ω - x_sample
                
                y_pred_all.append(y_pred)
            
            iters += 1
            if np.equal(np.array(y_pred_all), y).all():
                break
                
        self.iters, self.ω = iters, ω
        
    def predict(self, X):
        # Inject the bias column.
        X = np.asarray(X)
        X = np.concatenate((X, np.asarray([[1] * X.shape[0]]).T), axis=1)
        
        return (X @ self.ω > 0.5).astype(int)
clf = Perceptron()
clf.fit([[1], [2], [3]], [0, 0, 1])
clf.iters
clf.predict([[1], [2], [3]])
clf = Perceptron()
clf.fit([[1], [2], [3]], [0, 1, 0])
clf.iters
clf.predict([[1], [2], [3]])