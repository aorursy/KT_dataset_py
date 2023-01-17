%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import mpld3

mpld3.enable_notebook()



import warnings

warnings.simplefilter("always")
np.random.seed(0)

plt.plot(np.random.rand(10));
from mpld3 import plugins



fig, ax = plt.subplots(3, 3, figsize=(6, 6))

fig.subplots_adjust(hspace=0.1, wspace=0.1)

ax = ax[::-1]



X = np.random.normal(size=(3, 100))

for i in range(3):

    for j in range(3):

        ax[i, j].xaxis.set_major_formatter(plt.NullFormatter())

        ax[i, j].yaxis.set_major_formatter(plt.NullFormatter())

        points = ax[i, j].scatter(X[j], X[i])

        

plugins.connect(fig, plugins.LinkedBrush(points))
from mpld3 import plugins



fig, ax = plt.subplots()

ax.plot(np.random.random(10))

plugins.clear(fig)  # clear all plugins from the figure



plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom())
class HelloWorld(plugins.PluginBase):  # inherit from PluginBase

    """Hello World plugin"""

    

    JAVASCRIPT = """

    mpld3.register_plugin("helloworld", HelloWorld);

    HelloWorld.prototype = Object.create(mpld3.Plugin.prototype);

    HelloWorld.prototype.constructor = HelloWorld;

    function HelloWorld(fig, props){

        mpld3.Plugin.call(this, fig, props);

    };

    

    HelloWorld.prototype.draw = function(){

        this.fig.canvas.append("text")

            .text("hello world")

            .style("font-size", 72)

            .style("opacity", 0.3)

            .style("text-anchor", "middle")

            .attr("x", this.fig.width / 2)

            .attr("y", this.fig.height / 2)

    }

    """

    def __init__(self):

        self.dict_ = {"type": "helloworld"}

        

fig, ax = plt.subplots()

plugins.connect(fig, HelloWorld())