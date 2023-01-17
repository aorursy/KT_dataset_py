import numpy as np

import pandas as pd



from bokeh.resources import INLINE

import bokeh.io

bokeh.io.output_notebook(INLINE)

from bokeh.layouts import column, row

from bokeh.models import CustomJS, Slider, CDSView, Button, Label, GroupFilter, Arrow, OpenHead

from bokeh.plotting import ColumnDataSource, figure, output_file, show

from bokeh.palettes import RdYlBu11

from bokeh.transform import linear_cmap
filename = '/kaggle/input/mist-evolution-tracks/interpolatedStellarTracks.age7.447-7.505.steps200.pkl'

alldata = pd.read_pickle(filename)

#alldata = alldata.loc[alldata['star_age']]

modelAges = alldata.index.levels[1]

modelMasses = alldata.index.levels[0]

#print(list(zip(closeMass, modelAges)))



alldata.reset_index(inplace=True)



alldata['string_age'] = alldata.star_age.astype('str')



alldata['sizes'] = 10**alldata['log_R'] / 2 + 4

alldata.loc[alldata.sizes > 50, ['sizes']] = 50

alldata['sizes'] = alldata.sizes.round(0)

#alldata['sizes'] = alldata.sizes.astype('int')

#print(alldata['sizes'])



p = figure(plot_width=640, plot_height=480, x_range=[4.5, 3.2], y_range=[-3.5, 6.5],

           max_width=640, sizing_mode='scale_width',

          tooltips=[('Mass [Msun]', '@orig_mass'), ('log(L [Lsun])', '@log_L'),

                    ('log(Teff [K])', '@log_Teff'), ('log(R [Rsun])', '@log_R'),

                    ('Core H Frac', '@center_h1'), ('Core He Frac', '@center_he4'),

                    ('Phase', '@stage'),])



#for i in range(200):

#    print(str(modelAges[i]) == alldata.star_age.unique()[i])



data1 = alldata.loc[alldata.star_age <= modelAges[100]]

data2 = alldata.loc[alldata.star_age > modelAges[100]]



m9_1 = data1.loc[data1.orig_mass == 9.0]

m9_2 = data2.loc[data2.orig_mass == 9.0]

#m9 = alldata.loc[alldata.orig_mass == 9.0]

#print(m9.loc[alldata.string_age == str(modelAges[163])])

source = ColumnDataSource(alldata.loc[(alldata.orig_mass < 9) & (alldata.string_age == str(modelAges[0]))])

#m9src = ColumnDataSource(m9)

m9_1src = ColumnDataSource(m9_1)

m9_1src_copy = ColumnDataSource(m9_1)

m9_2src = ColumnDataSource(m9_2)

filt = GroupFilter(column_name='string_age', group=str(modelAges[0]))

#view1 = CDSView(source=m9src, filters=[filt])

view1 = CDSView(source=m9_1src, filters=[filt])

#view2 = CDSView(source=m9_2src, filters=[filt])

mapper = linear_cmap(field_name='log_Teff', palette=RdYlBu11 ,low=4.2 ,high=3.4)



p.circle(x='log_Teff', y='log_L', source=source, size='sizes', color=mapper)

#p.circle(x='log_Teff', y='log_L', source=m9src, view=view1, size='sizes', color=mapper)

p.circle(x='log_Teff', y='log_L', source=m9_1src, view=view1, size='sizes', color=mapper)

#p.circle(x='log_Teff', y='log_L', source=m9_2src, view=view2, size='sizes', color=mapper)

p.background_fill_color = "grey"

p.background_fill_alpha = 0.5

p.title.text = 'Age: {:1.4e} Yr'.format(modelAges[0])

p.title.align = 'center'

p.title.text_font_size = '16pt'

p.xaxis.axis_label = 'Log(T [K])'

p.yaxis.axis_label = 'Log(L [Lsun])'

p.xaxis.axis_label_text_font_size = '12pt'

p.yaxis.axis_label_text_font_size = '12pt'



slider = Slider(start=0, end=199, value=0, step=1, title="Model Number", max_width=620, sizing_mode='scale_width')

#slider2 = Slider(start=0, end=10, value=0, step=1, title="Model Number")

callback = CustomJS(args=dict(src1=m9_1src, src2=m9_2src, copysrc=m9_1src_copy, 

                              filt=filt, slider=slider, modelAges=modelAges, 

                              p=p),

                   code="""

    const pos = Math.round(slider.value);

    filt.group = String(modelAges[pos]);

    p.title.text = 'Age: ' + String(modelAges[pos].toExponential(4)) + ' Yr';

    if(pos > 100){

        src1.data = src2.data;

    }

    else{

        src1.data = copysrc.data;

    }

    // src2.change.emit();

    src1.change.emit();

                   """)

slider.js_on_change('value', callback)

layout = column(p, slider, sizing_mode='scale_width')







bokeh.io.show(layout)


filename = '/kaggle/input/mist-evolution-tracks/interpolatedStellarTracks.age7-8.steps100.pkl'

alldata = pd.read_pickle(filename)

#alldata = alldata.loc[alldata['star_age']]

modelAges = alldata.index.levels[1]

modelMasses = alldata.index.levels[0]

#print(list(zip(closeMass, modelAges)))



alldata = alldata.loc(axis=0)[modelMasses[::2]]



alldata.reset_index(inplace=True)

alldata['star_age'] = alldata.star_age.astype('str')



alldata['sizes'] = 10**alldata['log_R'] / 2 + 6

alldata.loc[alldata.sizes > 50, ['sizes']] = 50

alldata['sizes'] = alldata.sizes.round(0)

#alldata['sizes'] = alldata.sizes.astype('int')

#print(alldata['sizes'])



p = figure(plot_width=640, plot_height=480, x_range=[5.5, 3.0], y_range=[-3.5, 6.5],

           max_width=640, sizing_mode='scale_width',

          tooltips=[('Mass [Msun]', '@orig_mass'), ('log(L [Lsun])', '@log_L'),

                    ('log(Teff [K])', '@log_Teff'), ('log(R [Rsun])', '@log_R'),

                    ('Core H Frac', '@center_h1'), ('Core He Frac', '@center_he4'),

                    ('Phase', '@stage'),])





source = ColumnDataSource(alldata)

filt = GroupFilter(column_name='star_age', group=str(modelAges[0]))

view1 = CDSView(source=source, filters=[filt])

mapper = linear_cmap(field_name='log_Teff', palette=RdYlBu11 ,low=4.2 ,high=3.4)



p.circle(x='log_Teff', y='log_L', source=source, view=view1, size='sizes', color=mapper)

p.background_fill_color = "grey"

p.background_fill_alpha = 0.5

p.title.text = 'Age: {:1.4e} Yr'.format(modelAges[0])

p.title.align = 'center'

p.title.text_font_size = '16pt'

p.xaxis.axis_label = 'Log(T [K])'

p.yaxis.axis_label = 'Log(L [Lsun])'

p.xaxis.axis_label_text_font_size = '12pt'

p.yaxis.axis_label_text_font_size = '12pt'



slider = Slider(start=0, end=99, value=0, step=1, title="Model Number", max_width=620, sizing_mode='scale_width')

#slider2 = Slider(start=0, end=10, value=0, step=1, title="Model Number")

callback = CustomJS(args=dict(source=source, filt=filt, slider=slider, modelAges=modelAges, 

                              p=p),

                   code="""

    const pos = slider.value;

    filt.group = String(modelAges[pos]);

    p.title.text = 'Age: ' + String(modelAges[pos].toExponential(4)) + ' Yr';

    source.change.emit();

                   """)

slider.js_on_change('value', callback)

layout = column(p, slider, sizing_mode='scale_width',)



bokeh.io.show(layout)
alldata['B-V'] = -0.91 + 7090 / 10**alldata['log_Teff']

alldata['V'] = -2.5 * alldata['log_L'] + 3 # improve for next year

alldata['distance_modulus'] = 0.0

alldata['mag'] = alldata['V'] + alldata['distance_modulus']

alldata['colour'] = alldata['B-V'] - alldata['distance_modulus']



pleiades = np.loadtxt("/kaggle/input/cluster-data/pleiadesdata.csv",skiprows=5,usecols=(1,2), delimiter=',')

M13 = np.loadtxt("/kaggle/input/cluster-data/M13_BVdata.txt")

theory = alldata.loc[(alldata.star_age == str(modelAges[98])) & (alldata.stage == 'MS') & (alldata['B-V'] > -0.3)]





p = figure(plot_width=640, plot_height=480, x_range=[-0.5, 1.5], y_range=[22, 0],

           max_width=640, sizing_mode='scale_width')



p.xaxis.axis_label = 'B - V'

p.yaxis.axis_label = 'V'

p.xaxis.axis_label_text_font_size = '12pt'

p.yaxis.axis_label_text_font_size = '12pt'



pleiadesSource = ColumnDataSource(data=dict(x=pleiades[:,1] - pleiades[:,0], y=pleiades[:,0]))

M13Source = ColumnDataSource(data=dict(x=M13[:,0]-M13[:,1], y=M13[:,1]))

theorySource = ColumnDataSource(theory)



p.line(x='colour', y='mag', source=theorySource, color='black', line_width=15, alpha=0.5, legend_label='Theoretical MS')

p.square(x='x', y='y', source=pleiadesSource, size=7, legend_label = 'Pleiades')

p.circle(x='x', y='y', source=M13Source, size=1, color='orange', legend_label='M13')



p.legend.location = 'top_right'



slider = Slider(start=0, end=20, value=0, step=0.1, title="m - M", max_width=620, sizing_mode='scale_width')



callback = CustomJS(args=dict(source=theorySource, slider=slider),

                   code="""

    const pos = slider.value;

    for(var i=0; i<source.data['colour'].length; i++){

        source.data['mag'][i] = source.data['V'][i] + pos;

    }

    source.change.emit();

                   """)

slider.js_on_change('value', callback)



layout = column(p, slider, sizing_mode='scale_width')



bokeh.io.show(layout)
