import warnings  

import pandas as pd

import umap

from plotnine import *

warnings.filterwarnings('ignore')
df=pd.read_csv("../input/Iris.csv")

del df["Id"]
def clustUMAP(data,n_neighbors=list(range(5,55,10)),seg_col=None):

    """

    Creates UMAP projections of higher dimensional data

    Inputs:

        data- DataFrame to Vizualize

        n_neighbors- List of n_neighbors to try

        seg_col- Optional column with segment assignments

    Outputs:

        um

    """

    data=data.copy()

    if seg_col is not None:

        seg=data.pop(seg_col)

    plot_frame=pd.DataFrame()

    for n in n_neighbors:

        sub=umap.UMAP(n_neighbors=n).fit_transform(data)

        sub=pd.DataFrame(sub)

        sub.columns=["X","Y"]

        sub["run"]="N_Neighbors= "+str(n)

        if seg_col is not None:

            sub[seg_col]=seg

        plot_frame=plot_frame.append(sub)

    if seg_col is None:

        um=(ggplot(plot_frame,aes("X","Y"))+geom_point()+facet_wrap("run",scales = "free"))+labs(title="UMAP Plots",x="X",y="Y")

    else:

        um=(ggplot(plot_frame,aes("X","Y",color="factor("+seg_col+")"))+geom_point()+facet_wrap("run",scales = "free"))+labs(title="UMAP Plots",x="X",y="Y")

    return um
clustUMAP(df,n_neighbors=[25,30,35,40],seg_col="Species")