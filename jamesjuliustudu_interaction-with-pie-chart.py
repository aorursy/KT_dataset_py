import matplotlib.pyplot as plt
#import ipywidgets as wd
from ipywidgets import interactive
labels = ['Men', 'Female', 'Transgender'] 
chart_title = 'Gender distribution'
explodes = [0.1, 0.1, 0.4]
def piess(m, f, t):
    fig, ax=plt.subplots()    
    ax.pie([m,f,t], labels=labels, explode= explodes, autopct='%1.1f%%')
    plt.show()

interactive(piess, m=(0,45), f=(0,50), t=(0,5)) 
# wd.IntSlider(min=0, max=30, step=1, value=0)
