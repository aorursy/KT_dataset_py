%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


x = np.linspace(0, 0.5, 100)

plt.plot( x, 0.7 - 0.5*x + 0.3*np.exp(-x*20), label = "Overfitted model")
plt.plot( x, 0.9 - 0.5*x, label = "Non-Overfitted model")
plt.plot( x, 0.6 - 0.5*x, label = "Just Bad model")

axes = plt.gca()
axes.set_ylim([0, 1.1])

plt.legend(loc=3)
plt.suptitle("Expected decrease of accuracy in jitter test")

axes.set_xlabel('$\sigma$')
axes.set_ylabel('Accuracy')

plt.show()
from sklearn.metrics import accuracy_score

def jitter(X, scale):
    #out = X.copy()
    if scale > 0:        
        return X + np.random.normal(0, scale, X.shape)
    return X

def jitter_test(classifier, X, y, metric_FUNC = accuracy_score, sigmas = np.linspace(0, 0.5, 30), averaging_N = 5):
    out = []
    
    for s in sigmas:
        averageAccuracy = 0.0
        for x in range(averaging_N):
            averageAccuracy += metric_FUNC( y, classifier.predict(jitter(X, s)))

        out.append( averageAccuracy/averaging_N)

    return (out, sigmas, np.trapz(out, sigmas))

allJT = {}
from sklearn.ensemble import RandomForestClassifier
Y = [ 0 if z < np.median(x) else 1 for z in x ]
rf1 = RandomForestClassifier()
rf1.fit(x,Y)
jitter_test(rf1, x, Y)
