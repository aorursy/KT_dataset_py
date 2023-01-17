import pandas as pd



data = pd.read_csv('../input/data.csv')



data.head()
from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()

data.diagnosis = labelEncoder.fit_transform(data.diagnosis)

data.head()
import seaborn

import matplotlib.pyplot as matplot



graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'radius_mean', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'perimeter_mean', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'area_mean', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'smoothness_mean', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'compactness_mean', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'concavity_mean', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'concave points_mean', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'symmetry_mean', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'fractal_dimension_mean', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'texture_se', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'perimeter_se', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'area_se', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'smoothness_se', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'compactness_se', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'concavity_se', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'symmetry_se', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'fractal_dimension_se', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'radius_worst', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'texture_worst', bins=20)
graph = seaborn.FacetGrid(data, col='diagnosis')

graph.map(matplot.hist, 'perimeter_worst', bins=20)