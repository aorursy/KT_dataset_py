from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris

iris = load_iris()
# Euclidean distance.

def dist(a,b):

    return((sum(list(map(lambda x,y: (y-x)*(y-x),a,b))))**0.5)
def knn(data,target,point):

    k = int(len(data)**(0.5))

    distances = sorted(list(map(lambda x,y: [dist(x,point), y[0]], data, enumerate(data))))

    out = {x:0 for x in list(set(target))}

    for i in distances[:k]:

        out[target[i[1]]] += 1/(i[0]/min(distances)[0])

#        out[target[i[1]]] += 1

    return(max(out, key=out.get))
data, target = shuffle(iris.data, iris.target, random_state=9)

print("Sample size\tMy KNN\tsklearn's KNN")

for i in range(12,100):

    split = i

    traindata = data[:split]

    testdata = data[split:]

    traintarget = target[:split]

    testtarget = target[split:]

    prediction = list(map(lambda x: knn(traindata,traintarget,x), testdata))

    sklknn = KNeighborsClassifier(n_neighbors=int(i**(0.5)))

    sklknn.fit(traindata,traintarget)

    sklp = sklknn.predict(testdata)

    print(i,"\t\t",round(accuracy_score(testtarget,prediction),3),"\t",round(accuracy_score(testtarget,sklp),3))