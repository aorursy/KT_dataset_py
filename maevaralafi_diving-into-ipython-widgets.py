# Inline installation from within a notebook

# Uncomment these two lines if needed



#!pip install ipywidgets

#!jupyter nbextension enable --py --sys-prefix widgetsnbextension
import ipywidgets as widgets



# For explicitly displaying widgets

from IPython.display import display



# Just need these for the demo purposes here

from datetime import datetime

import matplotlib.pyplot as plt
# Create an simple IntSlider widget

demo_IntSlider_1 = widgets.IntSlider(

    min=1,                      # The minimum value

    max=100,                    # The maximum value

    description='Int Slider 1', # Label

    value=53,                   # Default value

)



# Display the widget

display(demo_IntSlider_1)
print("Demo Int Slider 1 Current Value:", demo_IntSlider_1.value)
# Create a simple IntSlider widget

demo_IntSlider_2 = widgets.IntSlider(

    min=1,                      # The minimum value

    max=100,                    # The maximum value

    description='Int Slider 2', # Label

    value=23,                   # Default value

)
# Create a simple FloatSlider widget

demo_FloatSlider_2 = widgets.FloatSlider(

    min=1,                      # The minimum value

    max=100,                    # The maximum value

    description='Float Slider 2', # Label

    value=64.87,                # Default value

)
# Display the widgets

display(demo_IntSlider_2)

display(demo_FloatSlider_2)
# Now, Synchronizing demo_IntSlider_2 and demo_FloatSlider_2

widgets.jslink(

    (demo_IntSlider_2, 'value'), 

    (demo_FloatSlider_2, 'value')

)



# Checkinng result

display(demo_IntSlider_2)

display(demo_FloatSlider_2)
# A basic button widget

demo_button_1 = widgets.Button(description='Basic Button')



# An event handler for the click on the button

def demo_button_eventhandler(obj):

    print('{} said: "Hello!"'.format(obj.description))



# Attaching event handler on the widget

demo_button_1.on_click(demo_button_eventhandler)



# Display the button

display(demo_button_1)
# A basic button widget

demo_button = widgets.Button(

    description='Basic Button',    # Label

    tooltip='Try to click on me',  # Tooltip caption that shows on hover

    icon='camera',                 # Optional font-awesome icon name

    disabled=False,                # Whether to disable user changes

)



# An event handler for the click on the button. The button will be passed as argument.

def demo_button_eventhandler(btn_obj):

    print('"Snapped a picture!"')



# Attaching event handler on the widget

demo_button.on_click(demo_button_eventhandler)



# Display the button

display(demo_button)
# Create the widget

demo_IntSlider = widgets.IntSlider(

    min=1,                      # The minimum value

    max=10,                     # The maximum value

    step=1,                     # The step change per move of the slide

    description='Int Slider',   # Label

    value=5,                    # Default value

    orientation='horizontal',   # Orientation of the slider

    readout=True,               # Display the current value of the slider next to it?

    readout_format=''           # Represent the format of the slider value for readout

)



# Display the widget

display(demo_IntSlider)
# Create the widget

demo_FloatSlider = widgets.FloatSlider(

    min=1,                      # The minimum value

    max=100,                    # The maximum value

    step=0.001,                 # The step change per move of the slide

    description='Float Slider', # Label

    value=25,                   # Default value

    orientation='vertical',     # Orientation of the slider

    readout=True,               # Display the current value of the slider next to it?

    readout_format='.3f'        # Represent the format of the slider value for readout

)



# Display the widget

display(demo_FloatSlider)
# List of options

state_options = ['', 'AL', 'AZ', 'CO', 'CA', 'FL', 'GE', 'MS', 'TN', 'WS']



# Create the widget

demo_Dropdown = widgets.Dropdown(

    options=state_options, # The list of available options

    index=None,             # The index of the default selection

    value='',               # The value of the default selection

    label='',               # The label corresponding to the selected value

    disabled=False,         # Whether to disable user changes

    description='State'    # Label

)



# Display the widget

display(demo_Dropdown)
# Create the widget

demo_Checkbox = widgets.Checkbox(

    description='Acknowledge Requirements', # Label

    value=False,                            # Default value

    disabled=False,                         # Whether to disable user changes

    indent=True                             # Align with other controls with a description

)



# Display the widget

display(demo_Checkbox)
# List of tuples as options

education_levels = [

    ('None', 0),

    ('High School', 1),

    ('Associate', 2),

    ('Bachelor', 3),

    ('Masters', 4),

    ('Doctorate', 5),

    ('Post-Doctorate', 6)

]



# Create the widget

demo_RadioButtons = widgets.RadioButtons(

    description='Education',       # Label

    options=education_levels,      # List of options or tuple of (label, value) for the dropdown

    disabled=False,                # Whether to disable user changes

    value=None,                    # Default value

    index=None,                    # The index of the default selection

    label=None                     # The label corresponding to the default selection

)



# Display the widget

display(demo_RadioButtons)
# Create the widget

demo_Text = widgets.Text(

    placeholder='e.g. John Smith', # Placeholder for text 

    description='Name',            # Label

    disabled=False                 # Whether to disable user changes

)



# Display the widget

display(demo_Text)
# Create the widget

demo_password = widgets.Password(

    placeholder='Enter password',  # Placeholder for text 

    description= 'Password',       # Label

    disabled=False                 # Whether to disable user changes

)



# Display the widget

display(demo_password)
# Create the widget

demo_DatePicker = widgets.DatePicker(

    value=datetime.now(),          # Default date selection

    description='Entry Date',      # Label

    disabled=False                 # Whether to disable user changes

)



# Display the widget

display(demo_DatePicker)
# Create the widget

demo_ColorPicker = widgets.ColorPicker(

    concise=False,                  # Whether to allow search by entering the name of the color or not

    description='Color',            # Label

    value='Green',                  # Default value

    disabled=False                  # Whether to disable user changes

)



# Display the widget

display(demo_ColorPicker)
# Create the widget

demo_IntProgress = widgets.IntProgress(

    min=1,                      # The minimum value

    max=100,                    # The maximum value

    description='Loading',      # Label

    value=35,                   # Default value

    orientation='horizontal',   # Orientation of the slider

    bar_style=''                # Color of the bar: 'success', 'info', 'warning', 'danger', ''

)



# Display the widget

display(demo_IntProgress)
# Create the widget

demo_FloatProgress = widgets.FloatProgress(

    min=1,                      # The minimum value

    max=100,                    # The maximum value

    description='Float Slider', # Label

    value=90,                   # Default value

    orientation='vertical',     # Orientation of the slider

    bar_style='danger'          # Color of the bar: 'success', 'info', 'warning', 'danger', ''

)



# Display the widget

display(demo_FloatProgress)
# Create the widget

demo_FileUpload = widgets.FileUpload(

    accept='.pdf',   # The type of files to accept

    multiple=True    # Whether to allow to upload multiple files or not

)



# Display the widget

display(demo_FileUpload)
for i, el in enumerate(dir(widgets)):

    print(str(i + 1) + '.', el)
import ipywidgets as widgets

from ipywidgets import interact



# Just need these for the demo purposes here

import numpy as np

import seaborn as sns
def plot_random_normal_distribution(n):

    """Generate random n number of data that follows a normal distibution and plot them on a histogram"""

    

    data = np.random.normal(size=n)         # Generate random data: n

    bins = int(np.sqrt(n))                  # Set bins for histogram: sqrt(n)

    sns.distplot(data, bins=bins, kde=True) # Plot histogram of data
# Creating a basic interctive plot

display(interact(

    plot_random_normal_distribution, # Function to call at any value changes

    n=10                             # Argument to pass to the function, with default value

))
# Using interact() as a decorator



@interact(

    # Specifying the handler for the n argument in the function

    n = widgets.IntSlider(min=2, max=100, step=1, value=10)

)

def plot_random_normal_distribution(n):

    """Generate random n number of data that follows a normal distibution and plot it on the graph"""

    

    data = np.random.normal(size=n)

    bins = int(np.sqrt(len(data)))

    sns.distplot(data, bins=bins, kde=True)

    

    # Adding some labels

    plt.title('Histogram of n random data')

    plt.xlabel('Random data')

    plt.ylabel('Proportion')
# Pre-specifying our handler widget

n_widget_handler = widgets.IntSlider(

    min=1,                      # The minimum value

    max=100,                    # The maximum value

    step=1,                     # The step change per move of the slide

    description='n',            # Label

    value=25,                   # Default value

    orientation='horizontal',   # Orientation of the slider

    readout=True                # Display the current value of the slider next to it?

)



# Specifying the handler for the n argument in the function

@interact(n = n_widget_handler)

def plot_random_scatterplot(n):

    """Generate random n number of data and plot n x n on a scatterplot"""

    

    x_data = np.random.random(size=n)

    y_data = np.random.random(size=n)

    sns.scatterplot(x=x_data, y=y_data)

    plt.title('Scatterplot of random data')

    plt.xlabel('Random x')

    plt.ylabel('Random y')