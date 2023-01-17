from IPython.display import IFrame

display(IFrame(src = 'https://www.theregister.co.uk/2019/11/07/python_java_github_javascript/', width=1000, height=700))
from IPython.display import YouTubeVideo

YouTubeVideo("HaSpqsKaRbo")
documentation = IFrame(src = 'https://ipywidgets.readthedocs.io/en/latest/index.html', width=1000, height=600)

display(documentation)
import ipywidgets

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
5 * 5
def f1(x):

    

    y = x * x

    print("{0} * {0} = {1}".format(x, y))
f1(3)
ipywidgets.interact(f1, x = 10);
@ipywidgets.interact(x = 10, y = 10)

def f2(x, y):

    z = x * y

    print("{0} * {1} = {2}".format(x, y, z))
def myPlot(frequency = 2, color = 'blue', lw = 4, grid = True):

    """

    plots cos(pi * f * x)

    """

    

    x = np.linspace(-3, 3, 1000)

    fig, ax = plt.subplots(1, 1, figsize = (6, 4))

    ax.plot(x, np.cos(np.pi * frequency * x), lw = lw, color = color)

    ax.grid(grid)

    plt.title("plot of cos(pi * f * x)", fontdict = {"size" : 20})

    

myPlot()
ipywidgets.interact(myPlot, color = ['blue', 'red', 'green'], lw = (1, 10));
ipywidgets.interact_manual(myPlot, color = ['blue', 'red', 'green'], lw = range(1, 10));
from sklearn.datasets import load_boston

boston = load_boston()



print(boston.DESCR)



boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)

boston_df["PRICE"] = boston.target

boston_df.head(10)
def filterColumn(column, threshold):

    

    boston_df_select = boston_df.loc[boston_df[column] > threshold]

    msg = "There are {:,} records for {} > {:,}".format(boston_df_select.shape[0], column, threshold)

    

    print("{0}\n{1}\n{0}".format("=" * len(msg), msg))

    display(boston_df_select.head(10))
filterColumn("RM", 5)
ipywidgets.interact(filterColumn, column = list(boston_df.columns), threshold = 30);
col_widget = ipywidgets.Select(

    description = " ",

    options = list(boston_df.columns),

    value = list(boston_df.columns)[0],

    layout = ipywidgets.Layout(

        width = '200px',

        height = '240px',

        margin = "0px",

        padding = "0px"))



threshold_widget = ipywidgets.FloatSlider(

    description = "Threshold: ",

    value = boston_df[col_widget.value].mean(),

    min = boston_df[col_widget.value].min(),

    max = boston_df[col_widget.value].max())



w1 = ipywidgets.interactive_output(filterColumn, {"column": col_widget, "threshold": threshold_widget})



ipywidgets.HBox(children = [col_widget, ipywidgets.VBox([threshold_widget, w1])], layout = ipywidgets.Layout(

        display = 'flex',

        flex_flow = 'row nowrap',

        align_items = 'flex-start',

        justify_content = 'center'))
@ipywidgets.interact

def calcCorr(column1 = boston_df.columns, column2 = boston_df.columns):

    print("Correlation between '{}' and '{}': {:.3f}".format(

        column1,

        column2,

        boston_df[column1].corr(boston_df[column2])))
w_bins = ipywidgets.BoundedIntText(description = "bins: ", value = 7, min = 5, max = 15, step = 1)

w_cols = ipywidgets.Select(description = "cols: ", 

                           options = boston_df.columns,

                           layout = ipywidgets.Layout(

                                width = '303px',

                                height = '240px',

                                margin = "0px",

                                padding = "0px"))



def plotHist(col, bins):

    

    boston_df[col].plot.hist(bins = bins);

    plt.xlabel(col)

    plt.title("Histogram")



w_hist = ipywidgets.interactive_output(plotHist, {"col": w_cols, "bins": w_bins})



ipywidgets.HBox([ipywidgets.VBox([w_bins, w_cols]), w_hist])
w_floatSlider = ipywidgets.FloatSlider(

    value = 7.5,

    min = 5.0,

    max = 10.0,

    step = 0.1,

    description = 'Input:')



w_floatSlider.style.handle_color = 'lightgreen'

w_floatSlider
w_floatSlider.value = 6.1
w_floatText = ipywidgets.FloatText(

    value = 2.4,

    step = 0.1,

    description = 'Value')



w_floatText
w_floatText.value
ipywidgets.link((w_floatSlider, 'value'), (w_floatText, 'value'))

ipywidgets.HBox([w_floatSlider, w_floatText])
w_datePicker = ipywidgets.DatePicker(description = "Pick a date: ")

w_datePicker
w_datePicker.value
w_start = ipywidgets.DatePicker(description = "Start date: ")

w_end = ipywidgets.DatePicker(description = "End date: ")

display(ipywidgets.VBox([w_start, w_end]))
diff = (w_end.value - w_start.value).days

print("There are {} days between '{}' and '{}'".format(diff, w_end.value, w_start.value))
w_colorPicker = ipywidgets.ColorPicker(

    description = "Pick a color: ",

    concise = False)



w_colorPicker
w_colorPicker.value
w_checkBox = ipywidgets.Checkbox(description = "check me, and I will return True")

w_checkBox
w_checkBox.value
w_radioButtons = ipywidgets.RadioButtons(

    options = ['Cappuccino', 'Espresso', 'Americano'],

    value = 'Cappuccino',

    description = 'Your Pick?')



w_radioButtons
w_radioButtons.value
w_select = ipywidgets.Select(

    options = ['Cappuccino', 'Espresso', 'Americano'],

    value = 'Cappuccino',

    description = 'Select:')



w_select
w_select.value
w_selectMultiple = ipywidgets.SelectMultiple(

    options = ['Cappuccino', 'Espresso', 'Americano'],

    value = ['Cappuccino'],

    description = 'Multi Select:')



w_selectMultiple
w_selectMultiple.value
w_toggleButtons = ipywidgets.ToggleButtons(

    options = ['Cappuccino', 'Espresso', 'Americano'],

    value = 'Cappuccino',

    button_style = 'info', # 'success', 'info', 'warning', 'danger' or ''

    tooltips = ['Description of Cappuccino', 'Description of Espresso', 'Description of Americano'])



w_toggleButtons
w_toggleButtons.value
w_button = ipywidgets.Button(

    description = "click me!",

    button_style = "info",

    tooltip = "This is your tooltip")



def onButtonClicked(change):

    tada = "\U0001F389"

    print ("{0}\n{1} button was clicked! {1}\n{0}".format(tada * 17, tada * 4))

    

w_button.style.button_color = 'brown'

w_button.style.font_weight = "bold"

w_button.on_click(onButtonClicked)

w_button
w_Text = ipywidgets.Text(

    placeholder = 'Enter your name...',

    description = "my app: ")



def onSubmit(change):

    print("Hello {}!".format(w_Text.value))



w_Text.on_submit(onSubmit)

w_Text
def celsiusToFahrenheit(temp):

    return "{:.2f}".format(1.8 * temp + 32)



def fahrenheitToCelsius(temp):

    return "{:.2f}".format((temp -32) / 1.8)



w_celsius = ipywidgets.FloatText(

    description = 'Celsius $^\circ$C',

    value = 0)



w_fahrenheit = ipywidgets.FloatText(

    description = 'Fahrenheit $^\circ$F', 

    value = celsiusToFahrenheit(w_celsius.value))



def onCelsiusChange(change):

    w_fahrenheit.value = celsiusToFahrenheit(change['new'])

    

def onFahrenheitChange(change):

    w_celsius.value = fahrenheitToCelsius(change['new'])

    

w_celsius.observe(onCelsiusChange, names = 'value')

w_fahrenheit.observe(onFahrenheitChange, names = 'value')



display(w_celsius, w_fahrenheit)
w_floatRangeSlider = ipywidgets.FloatRangeSlider(

    description = "RangeSlider:",

    value = [5, 15],

    min = 0,

    max = 20,

    step = 0.2, 

    orientation = "horizontal", # vertical

    readout = True,

    readout_format = "0.1f")



w_label = ipywidgets.Label(value = "Diff: 10")

display(w_label)



def onChangeRange(change):

    w_label.value = "Diff: {:.2f}".format(w_floatRangeSlider.upper - w_floatRangeSlider.lower)



w_floatRangeSlider.observe(onChangeRange)

w_floatRangeSlider
print("lower bound: {0.lower}\nupper bound: {0.upper}".format(w_floatRangeSlider))
w_password = ipywidgets.Password(

    description = "Enter pw: ",

    value = "password"

)

w_password
w_password.value
w_progressBar = ipywidgets.IntProgress(

    description = "loading...", 

    value = 0,

    min = 0,

    max = 100)

display(w_progressBar)



for i in range(100):

    w_progressBar.value = i

    time.sleep(0.04)

    

w_progressBar.description = "Done!"
w_html = ipywidgets.HTMLMath(

    value = r"""

        <b>Cappuccino is bold.</b> <br>

        <i>Espresso is italized.</i> <br>

        <u>Americano is underlined.</u> <br>

        <br>Some math: <br>

        $$i\hbar\frac{\partial}{\partial t} \Psi(\mathbf{r},t) = \left [ \frac{-\hbar^2}{2m}\nabla^2 + V(\mathbf{r},t)\right ] \Psi(\mathbf{r},t)$$""",

    description = 'HTML: ')



w_html
w_fileUpload = ipywidgets.FileUpload(

    accept = ".jpg",

    multiple = False)



w_fileUpload
class addon:

    

    def __init__(self, name):

        self.name = name

        

    def toadd(self):

        return ipywidgets.IntSlider(description = self.name, value = 0, min = 0, max = 3)



w_coffee = ipywidgets.RadioButtons(options = ['Cappuccino', 'Espresso', 'Americano'])

w_cup = ipywidgets.Select(options = ['Small', 'Medium', 'Large', 'Extra Large'])

w_suger = addon("Suger:").toadd()

w_milk = addon("Milk:").toadd()

w_cream = addon("Cream:").toadd()

w_smc = ipywidgets.HBox(children = [w_suger, w_milk, w_cream])

w_pay = ipywidgets.ToggleButtons(description = "Pay with?", options = ['Visa', 'MasterCard', 'AmEx'], button_style = "info")

w_finish = ipywidgets.Button(description = "Finish", button_style = "warning")

w_valid = ipywidgets.Valid(description = "All set!", value = True)

w_checkout = ipywidgets.HBox(children = [w_pay, w_finish], layout = ipywidgets.Layout(

        display = 'flex',

        flex_flow = 'row nowrap',

        align_items = 'flex-start',

        justify_content = 'space-between'))



w_tab = ipywidgets.Tab(children = [w_coffee, w_cup, w_smc, w_checkout])



def onButtonClicked(change):



    display(w_valid)



    print("You selected {} {} with {} suger, {} milk and {} cream; paying with {}.".format(

        w_cup.value, 

        w_coffee.value,

        w_suger.value,

        w_milk.value,

        w_cream.value,

        w_pay.value))



w_finish.on_click(onButtonClicked)



w_tab.set_title(0, 'Pick your coffee:')

w_tab.set_title(1, 'Pick your cup:')

w_tab.set_title(2, 'Suger/milk/cream:')

w_tab.set_title(3, 'Check-out:')

w_tab
w_accordion = ipywidgets.Accordion(children = [w_coffee, w_cup, w_smc, w_checkout])

w_accordion.set_title(0, 'Pick your coffee:')

w_accordion.set_title(1, 'Pick your cup:')

w_accordion.set_title(2, 'Suger/milk/cream:')

w_accordion.set_title(3, 'Check-out:')

w_accordion
w_audio = ipywidgets.Audio.from_url("https://www.kozco.com/tech/piano2-CoolEdit.mp3")

w_audio
w_video = ipywidgets.Video.from_url("https://www.radiantmediaplayer.com/media/bbb-360p.mp4")

w_video
w_selPic = ipywidgets.Select(options = ["fruits", "girl", "cat", "pool", "watch", "peppers"], 

        layout =ipywidgets.Layout(

            width = '120px',

            height = '150px',

            margin = "10px",

            padding = "10px"))



def disPic(url):

    url = "https://homepages.cae.wisc.edu/~ece533/images/" + url

    display(ipywidgets.Image.from_url(url))



w_disPic = ipywidgets.interactive_output(disPic, {"url": w_selPic})



ipywidgets.HBox([w_selPic, w_disPic], layout = ipywidgets.Layout(

        display = 'flex',

        flex_flow = 'row nowrap',

        align_items = 'flex-start',

        justify_content = 'center'))
import ipyleaflet



w_mapTypes = ipywidgets.Dropdown(

    options = [

        'Esri.WorldStreetMap',

        'Esri.DeLorme',

        'Esri.WorldTopoMap',

        'Esri.WorldImagery',

        'Esri.NatGeoWorldMap',

        'CartoDB.Positron',

        'CartoDB.DarkMatter',

        'HikeBike.HikeBike',

        'Hydda.Full',

        'NASAGIBS.ViirsEarthAtNight2012',

        'NASAGIBS.ModisTerraTrueColorCR',

        'OpenStreetMap.Mapnik',

        'OpenStreetMap.HOT',

        'OpenTopoMap',

        'Stamen.Toner',

        'Stamen.Watercolor'],

    value = 'Esri.WorldStreetMap',

    description = 'Map type:')





def toggleMap(mapType):

    m = ipyleaflet.Map(center = (43.6532, -79.3832), zoom = 6, basemap = eval("ipyleaflet.basemaps." + mapType))

    display(m)

    

w_map = ipywidgets.interactive_output(toggleMap, {"mapType": w_mapTypes});



ipywidgets.VBox([w_mapTypes, w_map])