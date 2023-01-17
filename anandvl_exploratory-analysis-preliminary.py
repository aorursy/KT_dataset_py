import numpy as Numpy
import pandas as Pandas
import matplotlib.pyplot as Plt
import seaborn as Sns
import os as Os
inpDir = '../input/'
outDir = ''
fileName = 'BlackFriday.csv'
fileTitle = fileName.split('.')[0]
bf = Pandas.read_csv(Os.path.join(inpDir + fileName), index_col=False)
bf.info()
Pandas.set_option('display.expand_frame_repr', False)
bf.describe()
print('Number of unique values (levels) for each column')
for cols in list(bf):
    print(f'{bf[cols].nunique()}: {cols}')
for cols in ['Product_Category_2', 'Product_Category_3']:
    print(f'{cols}: {sorted(bf[cols].unique())}')
    bf[cols].fillna(value=1, inplace=True)
    bf[cols] = bf[cols].apply(lambda x: int(x))
    print(f'{cols}: {sorted(bf[cols].unique())}')
fig1 = Plt.figure(1,figsize=(10,7.5))
fig1.clf()
k = 0
for cols in list(bf)[2:len(list(bf))-1]:    
    k = k + 1
    ax = fig1.add_subplot(3,3,k)
    
    bf_summ = bf[[cols,'Purchase']].groupby([cols])['Purchase'].agg(['sum','count']).reset_index().rename(columns={'sum':'totRev','count': 'totPurch'}).sort_values(['totRev'], ascending=[False])
    ax.pie(bf_summ['totRev'], labels = bf_summ[cols], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(cols)

fig1.subplots_adjust(hspace=0.5, wspace = 0.5)
fig1.savefig(Os.path.join(outDir + fileTitle + '_1.png'), dpi=300)
fig2 = Plt.figure(2,figsize=(7.5,10.0))
fig2.clf()
j = 0

fig3 = Plt.figure(3,figsize=(7.5,26.0))
fig2.clf()
k = 0

totRevenue = sum(bf.Purchase)
for cols in list(bf)[0:len(list(bf))-1]:
    # The following two statements could probably be combined and simplified using lambda functions in the '.agg' method.  Will do later.
    bf_summ = bf[[cols,'Purchase']].groupby([cols])['Purchase'].agg(['sum']).reset_index().rename(columns={'sum':'percRev'})

    bf_summ['percRev'] = 100.0*bf_summ['percRev']/totRevenue
    bf_summ = bf_summ.sort_values(['percRev'], ascending=[False]).copy()

    noFactors = bf_summ.shape[0]
    bf_summ['cumPercFactor'] = 100.0*(bf_summ.reset_index().index + 1)/noFactors
    
    bf_summ['cumPercRev'] = round(Numpy.cumsum(bf_summ['percRev']),2)
    bf_summ['percRev'] = round(bf_summ['percRev'],2)
    
    if (cols in ['User_ID', 'Product_ID']):
        j = j + 1
        ax = fig2.add_subplot(2,1,j)
        ax.plot(bf_summ['cumPercFactor'], bf_summ['cumPercRev'])
        ax.set_xlabel('Cummulative percentage of factors')
        ax.set_ylabel('Cummulative percentage of revenue')
        ax.set_title(cols)
        print(f'Cummulative Percentage of top {cols} that accounted for 80% of revenue: {round(bf_summ.iloc[max(max(Numpy.where(bf_summ["cumPercRev"] < 80.0)))]["cumPercFactor"], 1)}\n')
    else:
        k = k + 1
        ax = fig3.add_subplot(9,1,k)
        ax.bar(bf_summ[cols].apply(lambda x: str(x)), bf_summ['cumPercRev'])
        if (cols=='Age'):
            ax.set_xticklabels(bf_summ[cols], rotation='vertical', horizontalalignment="right")
        #ax.set_xlabel('Factors')
        ax.set_ylabel('Cummulative\n percentage\n of revenue')
        ax.set_title(cols)
        print(bf_summ.head(10))
        print('\n')
        
fig2.subplots_adjust(hspace=0.5, wspace = 0.0)
fig2.savefig(Os.path.join(outDir + fileTitle + '_2.png'), dpi=300)

fig3.subplots_adjust(hspace=0.5, wspace = 0.0)
fig3.savefig(Os.path.join(outDir + fileTitle + '_3.png'), dpi=300)