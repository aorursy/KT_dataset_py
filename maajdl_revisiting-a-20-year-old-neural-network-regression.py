import numpy as np
import pandas as pd
pd.options.display.float_format = '{:0.2f}'.format

from bokeh.layouts import row, column
from bokeh.plotting import figure, show
from bokeh.colors import RGB
from bokeh.models import Arrow, VeeHead, Label
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_notebook
output_notebook()

from IPython.display import display

import warnings
warnings.filterwarnings("ignore")
def plotSimple(title, xTitle, yTitle, x, y, y_range=(0, 100)):
    p = figure(title = title, plot_width=300, plot_height=250, y_range=y_range)
    p.circle(x, y, size=3, color="red", alpha=1)
    p.line(x, y, color="red", alpha=1)
    p.xaxis.axis_label = xTitle
    p.yaxis.axis_label = yTitle
    return p

def plotExperiment(xName, yName, experiment, df, model, blurcount, y_range=(0, 100)):
    def blur():
        lottery = np.random.random_sample(size=df.count()[0])
        lottery = [0.00 <= r < 0.70 for r in lottery]
        X = df[lottery][df.X]
        y = df[lottery][df.y]
        model.fit(X=X, y=y)
        Y = model.predict(experiment)
        p.line(experiment[xName], Y, color="lightblue",line_width=6, line_alpha=1)    
        p.line(experiment[xName], Y, color="blue",line_width=0, line_alpha=0.5)    
    p = figure(title = type(model).__name__, plot_width=300, plot_height=250, y_range=y_range)
    [blur() for i in range(blurcount)]
    Y = model.predict(X=experiment)
    p.line(experiment[xName], Y, color="red", line_width=3, line_alpha=1)    
    p.xaxis.axis_label = xName
    p.yaxis.axis_label = yName
    return p    
    
def plotRegression(df, model):  
    def plotRegression1(selection, color, alpha=1):
        X = df[selection][df.X]
        y = df[selection][df.y]
        Y = model.predict(X)
        r2 = np.corrcoef(y, Y)[0,1]
        se = np.std(y - Y)
        p.circle(y.tolist(), Y, size=3, color=color, alpha=alpha)
        p.line(y, y, color="red")
        return [r2,se]
    model.fit(X=df[df.train][df.X], y=df[df.train][df.y])    
    p = figure(title = type(model).__name__, plot_width=400, plot_height=400)
    trainr2, trainse = plotRegression1(df.train, "red")
    validr2, validse = plotRegression1(df.valid, "blue")
    testr2,  testse  = plotRegression1(df.test,  "green")
    r2text = "R2: train={r1: 6.2f}, valid={r2: 6.2f}, test={r3: 6.2f}".format(r1=trainr2,r2=validr2,r3=testr2)
    setext = "SE: train={r1: 6.2f}, valid={r2: 6.2f}, test={r3: 6.2f}".format(r1=trainse,r2=validse,r3=testse)
    p.text(x=[0.5], y = [0.2], text=[r2text], text_font_size="10pt")
    p.text(x=[0.5], y = [0.0], text=[setext], text_font_size="10pt")    
    p.xaxis.axis_label = "measured"
    p.yaxis.axis_label = "predicted"
    return p

def draw_PCAcomponents(df, ca, cb):
    def draw_component(df, feature, ca, cb):
        x_end=df[feature][ca]
        y_end=df[feature][cb]
        rl = (x_end**2+y_end**2)**0.5
        xl = x_end/rl * (1+rl)
        yl = y_end/rl * (1+rl)
        xl = x_end * 2
        yl = y_end * 2
        angle = np.arctan2(y_end,x_end)
        p.add_layout(Arrow(end=VeeHead(size=0), line_color="lightgrey", x_end=xl, y_end=yl))
        p.add_layout(Arrow(end=VeeHead(size=5), line_color="red", x_end=x_end, y_end=y_end))
        p.add_layout(Label(text = feature, x = xl, y = yl, text_font_size="8pt", text_color="blue", angle=angle))
    size = 1.5
    p = figure(title = "title", plot_width=400, plot_height=400, y_range=(-size,size), x_range=(-size,size))
    p.xaxis.axis_label = "C_" + str(ca)
    p.yaxis.axis_label = "C_" + str(cb)
    a = np.array([a*2*np.pi/36 for a in range(37)])
    [p.line(r*np.cos(a), r*np.sin(a), color="grey", alpha=r) for r in [0.2,0.4,0.6,0.8,1.0]]
    [draw_component(df, feature, ca, cb) for feature in df.columns.tolist()]
    return p

def myCorrelationPlot(df, pw=300, ph=300, tt="correlations"):
    colNames = df.columns.tolist()
    rownames = df.index.tolist()
    x = [c for r in rownames for c in colNames]
    y = [r for r in rownames for c in colNames]
    corarr = [df[c][r] for r in rownames for c in colNames]
    colors = [RGB(255*(1-x)/2,255*(1+x)/2,0,0.7) for x in corarr]
    p = figure(title=tt, x_range=colNames, y_range=rownames, plot_width=pw, plot_height=ph, toolbar_location="right")
    p.rect(x, y, color=colors, width=1, height=1)
    p.xaxis.major_label_orientation = 3.14159/2
    c = myColorBar(75, ph)
    return row(p,c)

def myColorBar(pw=75, ph=300, tt="colors"):
    from bokeh.models import CategoricalAxis
    corarr = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = [RGB(255*(1-x)/2,255*(1+x)/2,0,0.7) for x in corarr]
    colNames = ["color"]
    rownames = [str(x) for x in corarr]
    x = [c for r in rownames for c in colNames]
    y = [r for r in rownames for c in colNames]
    p = figure(title=tt, x_range=colNames, y_range=rownames, plot_width=pw, plot_height=ph, toolbar_location=None)
    p.rect(x, y, color=colors, width=1, height=1)
    p.xaxis.major_label_orientation = 3.14159/2
    return p

def panelTabs(ps):
    ts = [Panel(child=p, title=p.title.text) for p in ps]
    layout = Tabs(tabs=ts)
    show(layout)
url = "../input/Concrete_Data_Yeh.csv"
beton = pd.read_csv(url)
beton['w/c'] = beton['water']/beton['cement']
display(beton.describe())
show(myCorrelationPlot(beton.corr()))
from sklearn.decomposition import PCA
betonNorm2 = (beton-beton.mean())/beton.std()
pca = PCA().fit(betonNorm2)

var = pca.explained_variance_ratio_
varcum = [sum([w for w in var if w>=v]) for v in var]

p = plotSimple("cumulated variance","component","cum. var.", range(9), varcum, (0.2,1.1))
show(p)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
pd.options.display.float_format = '{:0.2f}'.format

pcadf = pd.DataFrame(pca.components_)
pcadf = pcadf.T
pcadf.columns = beton.columns

p1 = draw_PCAcomponents(pcadf, 0, 1)
p2 = draw_PCAcomponents(pcadf, 1, 2)
p3 = draw_PCAcomponents(pcadf, 2, 3)
p4 = draw_PCAcomponents(pcadf, 3, 4)
show(column(row(p1,p2),row(p3,p4)))
betonNorm = beton/beton.mean()
betonNorm.X = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']
betonNorm.y = 'csMPa'
lottery = np.random.random_sample(size=betonNorm.cement.count())
betonNorm.train = [0.00 <= r < 0.70 for r in lottery]
betonNorm.valid = [0.70 <= r < 0.85 for r in lottery]
betonNorm.test  = [0.85 <= r < 1.00 for r in lottery]
from sklearn import linear_model
linear = linear_model.LinearRegression()
p1 = plotRegression(betonNorm, linear)
importances = abs(linear.coef_)/np.linalg.norm(linear.coef_)
importances = pd.DataFrame(data=importances, columns=['importance'], index = betonNorm.X)
importances.sort_values(by="importance", ascending=False, inplace=True)
display(importances.T)
show(p1)
from sklearn import linear_model
ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
p1 = plotRegression(betonNorm, ransac)

inliers    = betonNorm[betonNorm.train]
inliers.y  = inliers[betonNorm.y]
inliers.X  = inliers[betonNorm.X]
outliers   = betonNorm[betonNorm.train][np.logical_not(ransac.inlier_mask_)]
outliers.y = outliers[betonNorm.y]
outliers.X = outliers[betonNorm.X]
outliers.p = ransac.predict(X=outliers.X)

from sklearn import linear_model
ransacInliers = linear_model.LinearRegression()
p2 = plotRegression(betonNorm, ransacInliers)
p2.circle(outliers.y, outliers.p, size=5, color="red", alpha=0.1)

importances = abs(ransac.estimator_.coef_)/np.linalg.norm(ransac.estimator_.coef_)
importances = pd.DataFrame(data=importances, columns=['importance'], index = betonNorm.X)
importances.sort_values(by="importance", ascending=False, inplace=True)
display(importances.T)
show(p2)
means = pd.DataFrame(inliers.mean(), columns=["inliers mean"])
means["outliers mean"] = outliers.mean()
display(means.T)
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(min_samples_split=10)
p1 = plotRegression(betonNorm, tree)

importances = pd.DataFrame(data=tree.feature_importances_, columns=['importance'], index = betonNorm.X)
importances.sort_values(by="importance", ascending=False, inplace=True)
display(importances.T)
show(p1)
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=10, random_state=0)
p1 = plotRegression(betonNorm, forest)

importances = pd.DataFrame(data=forest.feature_importances_, columns=['importance'], index = betonNorm.X)
importances.sort_values(by="importance", ascending=False, inplace=True)
display(importances.T)
show(p1)
from sklearn import svm
svr = svm.SVR(kernel='rbf', C=1)
p1 = plotRegression(betonNorm, svr)
show(row(p1))
from sklearn.neural_network import MLPRegressor
neural = MLPRegressor(solver='lbfgs', alpha=0.01, hidden_layer_sizes=(8), activation="logistic", max_iter=10000)
p1 = plotRegression(betonNorm, neural)  

importances = [max(abs(neural.coefs_[0][i])) for i in range(8)]
importances = pd.DataFrame(data=importances, columns=['importance'], index = betonNorm.X)
importances.sort_values(by="importance", ascending=False, inplace=True)
display(importances.T)
show(p1)
nsteps                = 50
experiment1                  = beton.head(nsteps).copy()
experiment1.cement           = beton.cement.mean()
experiment1.slag             = beton.slag.mean()
experiment1.flyash           = beton.flyash.mean()
experiment1.water            = beton.water.mean()
experiment1.superplasticizer = beton.superplasticizer.mean()
experiment1.coarseaggregate  = beton.coarseaggregate.mean()
experiment1.fineaggregate    = beton.fineaggregate.mean()
experiment1.age              = beton.age.mean()
experiment1.age              = 28
xxx                          = np.linspace(-50,50,nsteps)
experiment1.water            = experiment1.water  + xxx
experiment1.cement           = experiment1.cement - 3.15 * xxx
WCratio                      = experiment1.water / experiment1.cement
experiment1                  = experiment1/beton.mean()
experiment1                  = experiment1[betonNorm.X]

p1 = plotExperiment("water", "csMPa", experiment1, betonNorm, linear, 20, y_range=(0, 3))
p2 = plotExperiment("water", "csMPa", experiment1, betonNorm, ransac, 20, y_range=(0, 3))
p3 = plotExperiment("water", "csMPa", experiment1, betonNorm, tree, 20, y_range=(0, 3))
p4 = plotExperiment("water", "csMPa", experiment1, betonNorm, forest, 20, y_range=(0, 3))
p5 = plotExperiment("water", "csMPa", experiment1, betonNorm, svr, 20, y_range=(0, 3))
p6 = plotExperiment("water", "csMPa", experiment1, betonNorm, neural, 20, y_range=(0, 3))

show(column(row(p1, p2, p3), row(p4, p5, p6)))
panelTabs([p1,p2,p3,p4,p5,p6])
nsteps                = 50
experiment2                  = beton.head(nsteps).copy()
experiment2.cement           = beton.cement.mean()
experiment2.slag             = beton.slag.mean()
experiment2.flyash           = beton.flyash.mean()
experiment2.water            = beton.water.mean()
experiment2.superplasticizer = beton.superplasticizer.mean()
experiment2.coarseaggregate  = beton.coarseaggregate.mean()
experiment2.fineaggregate    = beton.fineaggregate.mean()
experiment2.age              = np.linspace(1,100,nsteps)
experiment2                  = experiment2/beton.mean()
experiment2                  = experiment2[betonNorm.X]

p1 = plotExperiment("age", "csMPa", experiment2, betonNorm, linear, 20, y_range=(0, 3))
p2 = plotExperiment("age", "csMPa", experiment2, betonNorm, ransac, 20, y_range=(0, 3))
p3 = plotExperiment("age", "csMPa", experiment2, betonNorm, tree, 20, y_range=(0, 3))
p4 = plotExperiment("age", "csMPa", experiment2, betonNorm, forest, 20, y_range=(0, 3))
p5 = plotExperiment("age", "csMPa", experiment2, betonNorm, svr, 20, y_range=(0, 3))
p6 = plotExperiment("age", "csMPa", experiment2, betonNorm, neural, 20, y_range=(0, 3))

show(column(row(p1, p2, p3), row(p4, p5, p6)))
panelTabs([p1,p2,p3,p4,p5,p6])
