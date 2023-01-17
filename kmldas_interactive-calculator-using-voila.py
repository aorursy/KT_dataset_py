from ipywidgets import HBox, VBox, IntSlider, interactive_output

from IPython.display import display



a = IntSlider()

b = IntSlider()



def f(a, b):

    print("{} * {} = {}".format(a, b, a * b))



out = interactive_output(f, { "a": a, "b": b })



display(HBox([VBox([a, b]), out]))
import ipywidgets as widgets



slider = widgets.FloatSlider(description='$x$', value=4)

text = widgets.FloatText(disabled=True, description='$x^2$')



def compute(*ignore):

    text.value = str(slider.value ** 2)



slider.observe(compute, 'value')



widgets.VBox([slider, text])
from ipywidgets import GridspecLayout, Button, BoundedIntText, Valid, Layout, Dropdown



def create_expanded_button(description, button_style):

    return Button(description=description, button_style=button_style, layout=Layout(height='auto', width='auto'))

 

rows = 11

columns = 6



gs = GridspecLayout(rows, columns)



def on_result_change(change):

    row = int(change["owner"].layout.grid_row)

    gs[row, 5].value = gs[0, 0].value * row == change["new"]

    

def on_multipler_change(change):

    for i in range(1, rows):

        gs[i, 0].description = str(change["new"])

        gs[i, 4].max = change["new"] * 10

        gs[i, 4].value = 1

        gs[i, 4].step = change["new"]

        gs[i, 5].value = False



gs[0, 0] = Dropdown(

    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    value=2,

)

gs[0, 0].observe(on_multipler_change, names="value")

multiplier = gs[0, 0].value



for i in range(1, rows):

    gs[i, 0] = create_expanded_button(str(multiplier), "")

    gs[i, 1] = create_expanded_button("*", "")

    gs[i, 2] = create_expanded_button(str(i), "info")

    gs[i, 3] = create_expanded_button("=", "")



    gs[i, 4] = BoundedIntText(

        min=0,

        max=multiplier * 10,

        layout=Layout(grid_row=str(i)),

        value=1,

        step=multiplier,

        disabled=False

    )



    gs[i, 5] = Valid(

        value=False,

        description='Valid!',

    )



    gs[i, 4].observe(on_result_change, names='value')



gs