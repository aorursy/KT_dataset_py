import networkx as nx
import pandas as pd
import numpy as np 
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns


korea_south=pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv',dtype=str)
region_ks=pd.read_csv('/kaggle/input/coronavirusdataset/Region.csv')

korea_south_region=pd.merge(korea_south,region_ks,on=['province','city'],how='inner')
korea_south_region['city_top_5']=np.where(korea_south_region.city.isin(['Cheonan-si','Bucheon-si',
                                                                            'Seongnam-si','Gyeongju-si',
                                                                            'Gunpo-si']),korea_south_region.city,'other_city')
korea_south_region['elementary_school_agg']=(korea_south_region['elementary_school_count']/20).apply(int)*20
korea_south_region['kindergarten_agg']=(korea_south_region['kindergarten_count']/20).apply(int)*20
korea_south_region['nursing_home_agg']=(korea_south_region['nursing_home_count']/500).apply(int)*500

korea_south_region_2=korea_south_region[korea_south_region['infected_by'].isin(korea_south_region['patient_id'])]
korea_south_region_2=korea_south_region_2[korea_south_region_2['infected_by']!='2000000205']
korea_south_region_2.shape


def create_network(korea_south_region_2,variable):
    #sns.palplot(current_palette)
    df=korea_south_region_2[~korea_south_region_2[variable].isnull()]
    #df.color.value_counts()
    G = nx.from_pandas_edgelist(df,'patient_id', 'infected_by',create_using=nx.Graph())#, edge_attr='age')
    #G.add_nodes_from(nodes_for_adding=df.patient_id.tolist())
    a=pd.DataFrame(G.nodes())
    a.columns=['patient_id']

    c=pd.merge(a,korea_south_region,on='patient_id',how='left')

    b=pd.DataFrame(c[variable].value_counts()).reset_index()
    b.columns=[variable,'freq']
    b=b.sort_values(variable)
    current_palette = sns.color_palette("bright",len(b))
    b['color']=current_palette

    #print(b)

    c=pd.merge(c,b,on=variable,how='left')

    c['color']=c['color'].fillna('grey')

    # Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!

    # Here is the tricky part: I need to reorder carac, to assign the good color to each node
    # Plot it, providing a continuous color scale with cmap:
    custom_lines =[]
    text_legend=[]
    for i in range(len(b)):
        custom_lines.append(Line2D([0], [0], color=b['color'].iloc[i], lw=2))
        text_legend.append(b[variable].iloc[i])

    fig, ax = plt.subplots()

    ax.legend(custom_lines, text_legend,fontsize=6)

    #nx.draw_networkx_labels(G,pos,labels,font_size=16)
    nx.draw(G, with_labels=False, node_color=c.color, node_size=4)#,alpha=0.8)
    #write_dot(G, 'file.dot')

    #optinonal for see it bigger 
    #plt.savefig(variable+"_south_korea.png",dpi=1200)
create_network(korea_south_region_2,'sex')
create_network(korea_south_region_2,'age')
create_network(korea_south_region_2,'city_top_5')
create_network(korea_south_region_2,'elementary_school_agg')
create_network(korea_south_region_2,'kindergarten_agg')
create_network(korea_south_region_2,'university_count')
create_network(korea_south_region_2,'nursing_home_agg')