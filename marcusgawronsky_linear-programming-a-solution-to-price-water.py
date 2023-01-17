! conda install -y -c conda-forge  hvplot=0.5.2 bokeh==1.4.0
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import holoviews as hv
from scipy.optimize import linprog
import hvplot.pandas  #noqa


hv.extension('bokeh')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
plant_names = ['alfalfa', 'corn', 'sorghum']
constraint_names = ['loam', 'sandy', 'alkaline', 'saline', 'crop rotation', 'water']

j = 4  # soil type
k = 3  # crop type
l = 3  # applied water-yield coefficient
m = 2  # water delivery

# planting
# X1 = np.array([])
# growing
# X2 = np.array([])
# harvesting
# x3 = np.array([])

# planting
## Assume fixed
# C1 = np.array([])

# labour
## $5/hour
## Assume fixed
# C2 = np.full(shape=(j, k, l, m), fill_value=5)

# revenue
## alfalfa, corn, sorghum
## https://www.sagis.org.za/conversion_table.html tonne = 39.3679 bu
## https://grains.org/markets-tools-data/tools/converting-grain-units/
C3 = np.array([[[73.38]*k,[2.75]*k,[2.46]*k]]*j)

# nature
P1 = np.array([0.25, 1-0.25]) #diversions
P2 = np.array([0.6, 1-0.6]) # precipitation
# soil type
R1 = np.array(
    [1700,  # loam
     9600,  # sandy soil
     16400,  # alkaline
     4500]  # aline
)
R1_1 = np.array([np.ones((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))])
R1_2 = np.array([np.zeros((3,3)), np.ones((3,3)), np.zeros((3,3)), np.zeros((3,3))])
R1_3 = np.array([np.zeros((3,3)), np.zeros((3,3)), np.ones((3,3)), np.zeros((3,3))])
R1_4 = np.array([np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.ones((3,3))])

# canal diversions
## acre-feet
R2_diversions = np.array([135000, 73100])

# inches [4, 8]
# acre feet [0.335, 0.67]
R2_precipitation = np.array([0.335, 0.67])*np.sum(R1)

ER = np.sum(P1 * R2_diversions + P2 * R2_precipitation)/2

# minimum crop rotation
## crop alfalfa, corn, sorghum
B_rot = np.array([[-5, 1, 1],  # loam
                  [-5, 1, 1],  # sandy soil
                  [-3, 1, 1],  # alkaline
                  [-3, 1, 1],  # saline
                 ])
B = np.repeat(np.expand_dims(B_rot, -1), l, -1)

# Irrigation Water Input Coefficients
## crop alfalfa, corn, sorghum
##  preplant
##   acre-inches [0, 4, 4]
##   acre-feet [0, 0.335, 0.335]
A1_preplant = np.array([0., 0.335, 0.335])
##  growing
##   acre-inches [0, 5*5, 4*3]
##   acre-feet [0., 0.42*5, 0.335*3]
A1_growing = np.array([0., 0.42*5., 0.335*3.])
##  harvest
##   acre-inches [6*4, 0, 0]
##   acre-feet [0.5*4, 0, 0]
A1_harvest = np.array([0.5*4, 0., 0.])
A1 = A1_preplant + A1_growing + A1_harvest
EA = np.repeat(np.expand_dims(A1, 0), l, 0) * np.full((j, k, l), 1)

# crop yields
## alfalfa, corn, sorghum
A2_yield_soil = np.array([[1, 1, 1],  # loam
               [1, 1-0.25, 1-0.1],  # sandy soil
               [0.8, 0.6, 0.7],  # alkaline
               [0.6, 0, 0],  # saline
              ])
## alfalfa, corn, sorghum
## corn 5 - 16 stalks per bushel, 18000-22000 stalks per care | 135
## https://www.uky.edu/ccd/sites/www.uky.edu.ccd/files/cornshocks.pdf
## https://mda.maryland.gov/farm_to_school/Documents/f2s_corn_math.pdf
## Alfalfa 86 bushels per acre in 2001, 140-160
## https://www.uaex.edu/publications/pdf/mp297/MP297.pdf
## https://www.hpj.com/dreiling/3-factors-to-boost-sorghum-yields/article_1213a4c5-4143-5abf-a4f3-b79671bb3e5d.html
A2_yield_acre = np.array([[4. * (1 + 3/4 + 2/4 + 1/4), 135, 150]])
A2 = A2_yield_soil * A2_yield_acre
def linear_programme(percent_of_water = 1):
    c = -np.array(C3.flatten()) * np.repeat(np.expand_dims(A2, -1), l, -1).flatten() # negative of objective function
    b = np.hstack((R1,
                   0, 
                   ER*percent_of_water)).reshape(-1, 1) # constraint bounds
    A = np.vstack((R1_1.flatten(),
                   R1_2.flatten(),
                   R1_3.flatten(),
                   R1_4.flatten(),
                   -B.flatten(),
                   EA.flatten()))  # constraints

    primal_result = linprog(c,A_ub=A, b_ub=b, options={'cholesky':False})

    dual_c = b
    dual_b = c
    dual_A = -A.T
    dual_result = linprog(dual_c, A_ub=dual_A, b_ub=dual_b, options={'cholesky':False})
    
    return (pd.DataFrame([primal_result.x.reshape((j, k, l)).sum((0, 2)).round(0)], 
                         columns=plant_names)
            .assign(Revenue = [-primal_result.fun])
            .assign(percent_of_water = percent_of_water)
            .assign(**dict(zip(['price ' + c for c in constraint_names], 
                               dual_result.x.round(0)))))

simulation_results = (pd.concat([linear_programme((p+1)/10) for p in range(10)], axis=0)
                      .assign(percent_of_water = lambda df: df.percent_of_water * 100))
(simulation_results.set_index('percent_of_water')
 .loc[:, plant_names]
 .hvplot(xlabel='Percent Expected Water (%)', ylabel='Acres Allocated',
         title='Optimal Allocation of Land under Water Restrictions'))
# which bounds are active
price_columns = (simulation_results
                 .columns[simulation_results
                          .columns
                          .str.startswith('price')])
(simulation_results
 .loc[:, ['percent_of_water'] + price_columns.tolist()]
 .melt(id_vars = 'percent_of_water', var_name='Constraint', value_name='Shadow Prices')
 .assign(Constraint = lambda df: df.Constraint.apply(lambda s: s.split()[1]))
 .replace({s: s.title() + ' (acre)' for s in constraint_names[:4]})
 .replace({'crop': 'Plant Rotation', 'water': 'Water  (feet/acre)'})
 .hvplot(x='percent_of_water', y='Shadow Prices', by='Constraint',
         ylabel = 'Implied (Shadow) Prices ($)',
         xlabel='Percent Expected Water (%)', title='Implied Price of Constraint, under water restrictions'))
simulation_results.hvplot.line(x='percent_of_water', y='Revenue',
                               ylabel='Revenue ($)', xlabel='Percent Expected Water (%)',
                               title='Revenue under varying Water Restrictions')