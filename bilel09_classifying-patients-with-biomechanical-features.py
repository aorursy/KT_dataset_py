import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')

data.head()
data.describe()
sns.countplot(data['class'])
plt.show()

data_melted = data.melt(id_vars=['class'])
ordered_class = data_melted["class"].value_counts().index
facet = sns.FacetGrid(data_melted, col="variable", sharey=False, col_wrap=2, aspect=1.2)
facet.map(sns.boxplot, "class", "value", data=data_melted, palette=["#e1812c", "#3a923a", "#3274a1"], order=ordered_class)
plt.show()
data = data[data.degree_spondylolisthesis<200]
sns.pairplot(data, hue="class", height=2.5)
plt.show()
corr = data.corr()
sns.heatmap(corr)
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=6)
x_data = data.iloc[:,0:-1]
pcs = pca.fit_transform(x_data)
plt.plot(np.arange(1,7),pca.explained_variance_ratio_ * 100)
plt.bar(np.arange(1,7),pca.explained_variance_ratio_ * 100)
plt.ylabel("Inertia (%)")
plt.xlabel("Dimension")
plt.show()
data_pca = np.hstack((data[['class']].to_numpy(), pcs))
data_pca = pd.DataFrame(data_pca)
data_pca = data_pca.rename(columns={i:f'Dim {i}' for i in range(1,7)}).rename(columns={0:'class'})

sns.scatterplot(x='Dim 1', y='Dim 2', hue='class', data= data_pca)
plt.show()

from mlxtend.plotting import plot_pca_correlation_graph

figure, correlation_matrix = plot_pca_correlation_graph(x_data, 
                                                        x_data.columns, 
                                                        dimensions=(1, 2), 
                                                        figure_axis_size=7, 
                                                        X_pca=pcs[:,0:2], 
                                                        explained_variance=pca.explained_variance_[0:2])

from sklearn.cluster import KMeans

data_pca_quant = data_pca.drop(columns="class")
km3 = KMeans (n_clusters=3, init="random")
km3.fit(data_pca_quant)
sns.scatterplot(x='Dim 1', y='Dim 2', data= data_pca_quant, hue=km3.labels_, style=data_pca['class'])
plt.show()
from sklearn.metrics import adjusted_rand_score

print(f"Adjusted rand score: {round(adjusted_rand_score(data_pca['class'], km3.labels_),2)}")
from sklearn.model_selection import train_test_split

X = data.iloc[:, 0:-1]
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42, stratify=y)
X_train_5c = X_train.drop(columns=['pelvic_incidence'])
X_test_5c = X_test.drop(columns=['pelvic_incidence'])

from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def grid_search(model, parameters):
    def one_grid_search(X, y, model, parameters):
        clf = GridSearchCV(model, parameters, scoring="accuracy", cv=5, n_jobs=-1, refit=True)
        clf.fit(X, y)
        scores = clf.cv_results_['mean_test_score']
        return clf.best_params_, clf.best_score_

    pipe1 = Pipeline([('standard', StandardScaler()), ('nca', NeighborhoodComponentsAnalysis())])
    pipe2 = Pipeline([('standard', StandardScaler()), ('nca', NeighborhoodComponentsAnalysis(2))])

    Xs = [ 
        ('original',X_train),
        ('without pelvic incidence',X_train_5c),

        ('NCA', NeighborhoodComponentsAnalysis().fit_transform(X_train,y_train)),
        ('without pelvic incidence + NCA', NeighborhoodComponentsAnalysis().fit_transform(X_train_5c,y_train)),
        
        ('NCA(2)', NeighborhoodComponentsAnalysis(2).fit_transform(X_train,y_train)),
        ('without pelvic incidence + NCA(2)', NeighborhoodComponentsAnalysis(2).fit_transform(X_train_5c,y_train)),
 
        ('standard scaler', StandardScaler().fit_transform(X_train)),
        ('without pelvic incidence + standard scaler', StandardScaler().fit_transform(X_train_5c)),
        
        ('standard scaler + NCA', pipe1.fit_transform(X_train, y_train)),
        ('without pelvic incidence + standard scaler + NCA', pipe1.fit_transform(X_train_5c, y_train)),

        ('standard scaler + NCA(2)', pipe2.fit_transform(X_train, y_train)),
        ('without pelvic incidence + standard scaler + NCA(2)', pipe2.fit_transform(X_train_5c, y_train))

    ]
    print(f'Parameters to test: {str(list(parameters.keys())).strip("[]")}\n')
    print(f"{'Method':55} {'Score':10} Parameters")
    for X in Xs:
        method, X = X
        best_params, best_score = one_grid_search(X, y_train, model, parameters)
        print(f'- {method:52}: ({round(100*best_score,2):5} %)  {str(list(best_params.values())).strip("[]")}')
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y):
    labels = ['Hernia', 'Spondylolisthesis', 'Normal']
    
    # Transfom y to categorical [1,2,3]
    y_cat = y.copy()
    y_cat[y_cat=='Hernia']=0
    y_cat[y_cat=='Spondylolisthesis']=1
    y_cat[y_cat=='Normal']=2
    y_cat = y_cat.to_numpy()
    y_cat = y_cat.astype(int)

    h = 20  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['tab:blue','tab:orange' , 'tab:green'])
    cmap_bold = ListedColormap(['blue', 'darkorange','darkgreen' ])

    # we create an instance of Neighbours Classifier and fit the data.
    clf.fit(X, y_cat)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 100, X[:, 0].max() + 100
    y_min, y_max = X[:, 1].min() - 100, X[:, 1].max() + 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_cat, cmap=cmap_bold,
                edgecolor='k', s=20, label=labels)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    legend = plt.legend(*scatter.legend_elements(), title="Class")
    for i, label in enumerate(labels):
        legend.get_texts()[i].set_text(label)
    plt.show()


from sklearn.neighbors import KNeighborsClassifier

parameters = {'n_neighbors':list(range(1, 100)), 'weights':['uniform', 'distance']}
grid_search(KNeighborsClassifier(n_jobs=-1), parameters)
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=14, weights='distance')
pipe = Pipeline([('standard', StandardScaler()), ('nca', NeighborhoodComponentsAnalysis(2))])
X_train_transformed = pipe.fit_transform(X_train, y_train)
X_test_transformed = pipe.transform(X_test)

knn.fit(X_train_transformed, y_train)
y_pred = knn.predict(X_test_transformed)
print(f'Test score: {round(100*accuracy_score(y_pred, y_test),2)} %')

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(knn, X_test_transformed, y_test, normalize='true')
plt.show()
knn = KNeighborsClassifier(n_neighbors=14, weights='distance')

plot_decision_boundary(knn, X_train_transformed, y_train)

from sklearn.linear_model import LogisticRegression

parameters = {'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}
grid_search(LogisticRegression(random_state=42, multi_class="auto", n_jobs=-1, C=1), parameters)
nca = NeighborhoodComponentsAnalysis(2)
X_train_transformed = nca.fit_transform(X_train_5c, y_train)
X_test_transformed = nca.transform(X_test_5c)

lr = LogisticRegression(random_state=42, multi_class="auto", n_jobs=-1, C=1, solver='newton-cg')
lr = lr.fit(X_train_transformed, y_train)

y_pred = lr.predict(X_test_transformed) 
print(f'Test score: {round(100*accuracy_score(y_pred, y_test),2)} %')


plot_confusion_matrix(lr, X_test_transformed, y_test, normalize='true')
plt.show()
lr = LogisticRegression(random_state=42, multi_class="auto", n_jobs=-1, C=1, solver='newton-cg')

plot_decision_boundary(lr, X_train_transformed, y_train)

from sklearn.ensemble import RandomForestClassifier

parameters = {
    'bootstrap':[True, False],            
    'criterion':['gini', 'entropy'], 
    'max_features':[2, 3, 4, 5, None],
    'n_estimators':[10, 100, 200],        
            }
grid_search(RandomForestClassifier(n_jobs=-1, random_state=42), parameters)

nca = NeighborhoodComponentsAnalysis()
X_train_transformed = nca.fit_transform(X_train, y_train)
X_test_transformed = nca.transform(X_test)

rf = RandomForestClassifier(bootstrap=False,criterion='gini', max_features=4, n_estimators=100,  random_state=42)
rf = rf.fit(X_train_transformed, y_train)

y_pred = rf.predict(X_test_transformed)
print(f'Test score: {round(100*accuracy_score(y_pred, y_test),2)} %')

plot_confusion_matrix(rf, X_test_transformed, y_test, normalize='true')
plt.show()