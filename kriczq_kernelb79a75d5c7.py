#mkjnksdjkns
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.collections as mcoll

import matplotlib.colors as mcolors

from matplotlib.widgets import Button

import json as js



TOLERANCE = 0.15



def dist(point1, point2):

    return np.sqrt(np.power(point1[0] - point2[0], 2) + np.power(point1[1] - point2[1], 2))



class _Button_callback(object):

    def __init__(self, scenes):

        self.i = 0

        self.scenes = scenes

        self.adding_points = False

        self.added_points = []

        self.adding_lines = False

        self.added_lines = []

        self.adding_rects = False

        self.added_rects = []



    def set_axes(self, ax):

        self.ax = ax

        

    def next(self, event):

        self.i = (self.i + 1) % len(self.scenes)

        self.draw(autoscaling = True)



    def prev(self, event):

        self.i = (self.i - 1) % len(self.scenes)

        self.draw(autoscaling = True)

        

    def add_point(self, event):

        self.adding_points = not self.adding_points

        self.new_line_point = None

        if self.adding_points:

            self.adding_lines = False

            self.adding_rects = False

            self.added_points.append(PointsCollection([]))

         

    def add_line(self, event):   

        self.adding_lines = not self.adding_lines

        self.new_line_point = None

        if self.adding_lines:

            self.adding_points = False

            self.adding_rects = False

            self.added_lines.append(LinesCollection([]))



    def add_rect(self, event):

        self.adding_rects = not self.adding_rects

        self.new_line_point = None

        if self.adding_rects:

            self.adding_points = False

            self.adding_lines = False

            self.new_rect()

    

    def new_rect(self):

        self.added_rects.append(LinesCollection([]))

        self.rect_points = []

    

    def on_click(self, event):

        if event.inaxes != self.ax:

            return

        new_point = (event.xdata, event.ydata)

        if self.adding_points:

            self.added_points[-1].add_points([new_point])

            self.draw(autoscaling = False)

        elif self.adding_lines:

            if self.new_line_point is not None:

                self.added_lines[-1].add([self.new_line_point, new_point])

                self.new_line_point = None

                self.draw(autoscaling = False)

            else:

                self.new_line_point = new_point

        elif self.adding_rects:

            if len(self.rect_points) == 0:

                self.rect_points.append(new_point)

            elif len(self.rect_points) == 1:

                self.added_rects[-1].add([self.rect_points[-1], new_point])

                self.rect_points.append(new_point)

                self.draw(autoscaling = False)

            elif len(self.rect_points) > 1:

                if dist(self.rect_points[0], new_point) < (np.mean([self.ax.get_xlim(), self.ax.get_ylim()])*TOLERANCE):

                    self.added_rects[-1].add([self.rect_points[-1], self.rect_points[0]])

                    self.new_rect()

                else:    

                    self.added_rects[-1].add([self.rect_points[-1], new_point])

                    self.rect_points.append(new_point)

                self.draw(autoscaling = False)

        

    def draw(self, autoscaling = True):

        if not autoscaling:

            xlim = self.ax.get_xlim()

            ylim = self.ax.get_ylim()

        self.ax.clear()

        for collection in (self.scenes[self.i].points + self.added_points):

            if len(collection.points) > 0:

                self.ax.scatter(*zip(*(np.array(collection.points))), **collection.kwargs)

        for collection in (self.scenes[self.i].lines + self.added_lines + self.added_rects):

            self.ax.add_collection(collection.get_collection())

        self.ax.autoscale(autoscaling)

        if not autoscaling:

            self.ax.set_xlim(xlim)

            self.ax.set_ylim(ylim)

        plt.draw()

class Scene:

    def __init__(self, points=[], lines=[]):

        self.points=points

        self.lines=lines



class PointsCollection:

    def __init__(self, points, **kwargs):

        self.points = points

        self.kwargs = kwargs

    

    def add_points(self, points):

        self.points = self.points + points



class LinesCollection:

    def __init__(self, lines, **kwargs):

        self.lines = lines

        self.kwargs = kwargs

        

    def add(self, line):

        self.lines.append(line)

        

    def get_collection(self):

        return mcoll.LineCollection(self.lines, **self.kwargs)

    

class Plot:

    def __init__(self, scenes = [Scene()], json = None):

        if json is None:

            self.scenes = scenes

        else:

            self.scenes = [Scene([PointsCollection(pointsCol) for pointsCol in scene["points"]], 

                                 [LinesCollection(linesCol) for linesCol in scene["lines"]]) 

                           for scene in js.loads(json)]

        

    def __configure_buttons(self):

        plt.subplots_adjust(bottom=0.2)

        ax_prev = plt.axes([0.6, 0.05, 0.15, 0.075])

        ax_next = plt.axes([0.76, 0.05, 0.15, 0.075])

        ax_add_point = plt.axes([0.44, 0.05, 0.15, 0.075])

        ax_add_line = plt.axes([0.28, 0.05, 0.15, 0.075])

        ax_add_rect = plt.axes([0.12, 0.05, 0.15, 0.075])

        b_next = Button(ax_next, 'Następny')

        b_next.on_clicked(self.callback.next)

        b_prev = Button(ax_prev, 'Poprzedni')

        b_prev.on_clicked(self.callback.prev)

        b_add_point = Button(ax_add_point, 'Dodaj punkt')

        b_add_point.on_clicked(self.callback.add_point)

        b_add_line = Button(ax_add_line, 'Dodaj linię')

        b_add_line.on_clicked(self.callback.add_line)

        b_add_rect = Button(ax_add_rect, 'Dodaj figurę')

        b_add_rect.on_clicked(self.callback.add_rect)

        return [b_prev, b_next, b_add_point, b_add_line, b_add_rect]

    

    def add_scene(self, scene):

        self.scenes.append(scene)

    

    def add_scenes(self, scenes):

        self.scenes = self.scenes + scenes



    def toJson(self):

        return js.dumps([{"points": [np.array(pointCol.points).tolist() for pointCol in scene.points], 

                          "lines":[linesCol.lines for linesCol in scene.lines]} 

                         for scene in self.scenes])    

     

    def get_added_points(self):

        if self.callback:

            return self.callback.added_points

        else:

            return None

  

    def get_added_lines(self):

        if self.callback:

            return self.callback.added_lines

        else:

            return None

        

    def get_added_figure(self):

        if self.callback:

            return self.callback.added_rects

        else:

            return None

    

    def get_added_elements(self):

        if self.callback:

            return Scene(self.callback.added_points, self.callback.added_lines+self.callback.added_rects)

        else:

            return None

    

    def draw(self):

        plt.close()

        fig = plt.figure()

        self.callback = _Button_callback(self.scenes)

        self.widgets = self.__configure_buttons()

        ax = plt.axes(autoscale_on = False)

        self.callback.set_axes(ax)

        fig.canvas.mpl_connect('button_press_event', self.callback.on_click)

        plt.show()

        self.callback.draw()

        
%matplotlib notebook



scenes=[Scene([PointsCollection([(1, 2), (3, 1.5), (2, -1)]), 

               PointsCollection([(5, -2), (2, 2), (-2, -1)], color='green', marker = "^")], 

              [LinesCollection([[(1,2),(2,3)], [(0,1),(1,0)]])]), 

        Scene([PointsCollection([(1, 2), (3, 1.5), (2, -1)], color='red'), 

               PointsCollection([(5, -2), (2, 2), (-2, 1)], color='black')], 

              [LinesCollection([[(-1,2),(-2,3)], [(0,-1),(-1,0)]])])]



plot = Plot(scenes)

plot.draw() 

scenes=[Scene([PointsCollection([(-2, -1)], color='green', marker = "^")], 

              [LinesCollection([[(1,2),(2,3)], [(0,1),(1,0)]])]), 

        Scene([PointsCollection([(1, 2), (3, 1.5)], color='red'), 

               PointsCollection([(5, -2)], color='black')])]

plot = Plot(scenes)



with open('somefile.json', 'w') as file:

    file.write(plot.toJson())

    

#somefile.txt: [{"points": [[[-2, -1]]], "lines": [[[[1, 2], [2, 3]], [[0, 1], [1, 0]]]]}, {"points": [[[1.0, 2.0], [3.0, 1.5]], [[5, -2]]], "lines": []}]

    

with open('somefile.json', 'r') as file:

    json = file.read()

    

plot2 = Plot(json=json)

plot2.draw()
#Dodaj tu punkty, figury i/lub odcinki!

plot1 = Plot()

plot1.draw()
plot2 = Plot([plot1.get_added_elements()])

plot2.draw()
%matplotlib notebook


