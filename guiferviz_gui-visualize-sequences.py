
import sys, os, random

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np

import pandas as pd

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


TRAIN_DATA_PATH = "input/train.csv"


def map_str2array(data, sep=","):
    return data.map(lambda x: np.array(x.split(sep), dtype=float))


class ShowSampleForm(QMainWindow):

    def __init__(self, parent=None):
        self.load_data()

        # Create GUI.
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Show sequences')
        self.create_main_frame()
        self.create_status_bar()

        self.on_draw()

    def load_data(self):
        try:
            # Load data from TRAIN_DATA_PATH.
            self.train = pd.read_csv(TRAIN_DATA_PATH)
            self.train.columns = ['id', 'seq']
            self.train['seq'] = map_str2array(self.train['seq'])
        except IOError:
            # No data file found, shown an error and exit.
            flags = QMessageBox.Close
            QMessageBox.critical(None, "Error",
                "No %s file found :'(" % TRAIN_DATA_PATH, flags)
            sys.exit()

    def on_draw(self):
        # Clear the axes and redraw the plot anew.
        self.axes.clear()

        # Prepare sample.
        sample = self.spinbox.value()
        sample_id = self.train['id'][sample]
        self.axes.set_title("Sample %d (id %d)" % (sample, sample_id))

        # Prepare (x, y) data to plot.
        y = self.train['seq'][sample]
        x = np.linspace(1, len(y), len(y))

        # Show last point?
        if self.last_cb.isChecked():
            self.axes.scatter(x, y, s=20, c='green', alpha=0.7)
            self.axes.plot(x, y, c='gray', alpha=0.7)
            for j in range(4):
                self.axes.scatter(x[-1], y[-1], s=10+5**j, c='darkblue', alpha=0.52-0.12*j)
        else:
            self.axes.scatter(x[:-1], y[:-1], s=20, c='green', alpha=0.7)
            self.axes.plot(x[:-1], y[:-1], c='gray', alpha=0.7)

        # Show axis?
        if self.axis_cb.isChecked():
            self.axes.axis("on")
        else:
            self.axes.axis("off")
            self.axes.set_xticks([])
            self.axes.set_yticks([])

        # Adjust axes.
        self.axes.relim()
        self.axes.autoscale()
        self.fig.tight_layout()

        # Redraw.
        self.canvas.draw()

    def create_main_frame(self):
        self.main_frame = QWidget()
        
        # Create the mpl Figure and FigCanvas objects.
        self.fig = Figure((5.5, 4.0), dpi=100)
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        
        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        
        # Other GUI controls
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(self.train.shape[0] - 1)
        self.spinbox.setValue(0)
        self.spinbox.setMinimumWidth(100)
        self.spinbox.setMaximumWidth(100)
        self.connect(self.spinbox, SIGNAL('valueChanged(int)'), self.on_draw)
        
        self.axis_cb = QCheckBox("Show &Axis")
        self.axis_cb.setChecked(False)
        self.connect(self.axis_cb, SIGNAL('stateChanged(int)'), self.on_draw)

        self.last_cb = QCheckBox("Show &Last")
        self.last_cb.setChecked(True)
        self.connect(self.last_cb, SIGNAL('stateChanged(int)'), self.on_draw)

        # Layout with box sizers.
        hbox = QHBoxLayout()
        hbox.addWidget(self.spinbox)
        hbox.addWidget(self.axis_cb)
        hbox.addWidget(self.last_cb)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)
        
        # Set main layout.
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def create_status_bar(self):
        self.status_text = QLabel("Visualize sequences in the training set.")
        self.statusBar().addWidget(self.status_text)


def main():
    app = QApplication(sys.argv)
    form = ShowSampleForm()
    form.show()
    app.exec_()


if __name__ == "__main__":
    print("Uncomment the line below. Call main function on your computer :)")
    #main()
