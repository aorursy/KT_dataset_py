import pandas as pd

main_file_path = '../input/clinvar_conflicting.csv' # this is the path to the genetic data that you will use
GeneData = pd.read_csv(main_file_path)
GeneData.describe() #describes the columns of eac data in terms of the count, mean, standard deviation, minimum, lower quartile, median, upper quartile and maximum
GeneData.CLASS #data that describes whether a patient has conflicting (1) or consistent (0) classification
GeneData.head() #displays the first five rows
from plotnine import * #plotline graphs
GD=GeneData.head(1000) #Variable stores the first rows  

(
    ggplot(GD)
        + aes('AF_EXAC', 'AF_TGP')
        + geom_point()
        + stat_smooth()
)
#plots a line of best fit (logistic regression) along the scatter graph to find the correlation between AF_EXAC and AF_TGP
from plotnine import * #plotline graphs

#shows the most common amino acids among different POS
(ggplot(GeneData.head(50))
         + aes('POS', 'Amino_acids')
         + geom_bin2d(bins=20)
         + ggtitle("Most Common amino acids")
)
#The plotnine equivalent of a hexplot, a two-dimensional histogram, is geom_bin2d
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go #for plotly graphs
dtt = GeneData.head(800).assign(n=0).groupby(['CADD_RAW', 'CADD_PHRED'])['n'].count().reset_index()
dtt = dtt[dtt["CADD_PHRED"] < 2000]
ver = dtt.pivot(index='CADD_PHRED', columns='CADD_RAW', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=ver)])
#plotly Surface (the most impressive feature)
#shows the distribution of CADD_RAW against CADD_PHRED
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go #for plotly graphs
dtt = GeneData.head(800).assign(n=0).groupby(['POS', 'CADD_PHRED'])['n'].count().reset_index()
dtt = dtt[dtt["CADD_PHRED"] < 2000]
ver = dtt.pivot(index='CADD_PHRED', columns='POS', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=ver)])
#plotly Surface (the most impressive feature)
#shows the distribution of POS against CADD_PHRED
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go #for plotly graphs
dtt = GeneData.head(800).assign(n=0).groupby(['POS', 'CADD_RAW'])['n'].count().reset_index()
dtt = dtt[dtt["CADD_RAW"] < 2000]
ver = dtt.pivot(index='CADD_RAW', columns='POS', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=ver)])
#plotly Surface (the most impressive feature)
#shows the distribution of POS against CADD_RAW
GeneData['CLASS'].plot.hist() #plots a hisogram for all the data in SalePrice
import seaborn as sns

sns.lmplot(x='POS', y='AF_TGP', hue='CLNDN', 
           data=GeneData.loc[GeneData['CLNDN'].isin(['Immunodeficiency_14|not_specified', 'Idiopathic_generalized_epilepsy|not_specified', 'Nephronophthisis|Renal_dysplasia_and_retinal_aplasia|not_specified'])], 
           fit_reg=False)
#multivariate scatter plot
Diseasesubmission = pd.DataFrame({'Position of variant (POS)': GeneData.POS,'allele frequency (AF_TGP)':GeneData.AF_TGP, 'disease': GeneData.CLNDN})

Diseasesubmission.to_csv('DiseasesSubmission.csv', index=False)