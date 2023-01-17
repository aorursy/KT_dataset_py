from IPython.display import HTML

import warnings

warnings.filterwarnings('ignore')

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/f-snXe2Bn9Q?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
show_widget = False
from clustergrammer2 import net as net

df = {}

if show_widget == False:

    print('\n-----------------------------------------------------')

    print('>>>                                               <<<')    

    print('>>> Please set show_widget to True to see widgets <<<')

    print('>>>                                               <<<')    

    print('-----------------------------------------------------\n')    

    delattr(net, 'widget_class') 
net.load_file('../input/pbmc_2700_cell_types.txt')

df = net.export_df()

df.index = [(x[0], 'Cell Type: ' + x[1]) for x in df.index.tolist()]

df.shape
cat_colors = net.load_json_to_dict('../input/cell_type_colors.json')

net.load_df(df)

net.set_cat_colors(cat_colors, 'col', 1, cat_title='Cell Type')

net.set_cat_colors(cat_colors, 'row', 1, cat_title='Cell Type')
net.filter_N_top(inst_rc='row', N_top=250, rank_type='var')

net.normalize(axis='row', norm_type='zscore')

net.load_df(net.export_df().round(2))

net.clip(lower=-5, upper=5)

net.widget()