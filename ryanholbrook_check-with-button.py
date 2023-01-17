# Setup feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.python.ex1 import *



import ipywidgets as widgets
check_q1 = widgets.Button(

    description="Check Question 1",

    tooltip="Click me!",

    icon="check",

)

output = widgets.Output()



def check_cb(button):

    output.clear_output()

    with output:

        q1.check()



check_q1.on_click(check_cb)
radius, area = [3/2, (3/2)**2 * 3.0]

widgets.VBox([check_q1, output])