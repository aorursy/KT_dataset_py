!pip install colour-science
import colour
import numpy
import matplotlib
matplotlib.pyplot.style.use({'figure.figsize': (8, 8)})
LMS_CMFS = colour.CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"]

# colour.plotting.plot_multi_cmfs(cmfs=["Stockman & Sharpe 2 Degree Cone Fundamentals"])
import ipywidgets
import IPython

wavelength_range = LMS_CMFS.shape.end - LMS_CMFS.shape.start
middle_wavelength = numpy.round(
    (wavelength_range / 2.0) + LMS_CMFS.shape.start
)

widget_XYZ_label = ipywidgets.widgets.Label(
    value="XYZ: "
)

widget_wavelength = ipywidgets.widgets.FloatSlider(
    value=middle_wavelength,
    min=LMS_CMFS.shape.start,
    max=LMS_CMFS.shape.end,
    step=LMS_CMFS.shape.interval,
    continuous_update=False,
    layout=ipywidgets.Layout(width="50%")
)

widget_fwhm = ipywidgets.widgets.FloatSlider(
    value=25.0,
    min=1.0,
    max=700.0,
    step=1.0,
    continuous_update=False,
    layout=ipywidgets.Layout(width="50%")
)


widget_graph1 = ipywidgets.widgets.Output()
widget_graph2 = ipywidgets.widgets.Output()
widget_graph3 = ipywidgets.widgets.Output()

widget_hbox = ipywidgets.widgets.HBox(
    [
        widget_graph1,
        widget_graph2,
        widget_graph3
    ]
)

widget_vbox = ipywidgets.widgets.VBox(
    [
        widget_wavelength,
        widget_fwhm,
        widget_XYZ_label,
        widget_hbox
    ]
)

def interaction(wavelength, fwhm):
    widget_XYZ_label.value = "XYZ: ".format(0)
    
    widget_graph1
    widget_graph2
    widget_graph3
    
def widget_graphs_update(change):
    widget_graph1.clear_output()
    widget_graph2.clear_output()
    widget_graph3.clear_output()

    sd_gaussian = colour.sd_gaussian(
        middle_wavelength,
        widget_fwhm.value,
        shape=LMS_CMFS.shape
    )
    
    with widget_graph1:
        sd_gaussian.values = numpy.roll(
            sd_gaussian.values,
            int(widget_wavelength.value - middle_wavelength)
        )

        colour.plotting.plot_single_sd(
            sd_gaussian,
            cmfs=LMS_CMFS,
            modulate_colours_with_sd_amplitude=True
        )
    
    with widget_graph2:
        colour.plotting.plot_sds_in_chromaticity_diagram_CIE1931(
        [sd_gaussian],
        cmfs=LMS_CMFS
        )
        
    with widget_graph3:
        colour.plotting.plot_sds_in_chromaticity_diagram_CIE1976UCS(
        [sd_gaussian],
        cmfs=LMS_CMFS
        )

widget_wavelength.observe(widget_graphs_update, "value")
widget_fwhm.observe(widget_graphs_update, "value")

widget_vbox