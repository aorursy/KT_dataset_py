!pip install -q pandapower
import pandapower as pp

import pandapower.networks as nw

import pandapower.plotting as plot



%matplotlib inline
net = nw.mv_oberrhein()

net
bc = plot.create_bus_collection(net, 

                                net.bus.index, 

                                size=80, 

                                color="red", 

                                zorder=10)



lcd = plot.create_line_collection(net, 

                                  lines=net.line.index, 

                                  color="grey",

                                  alpha=0.8, 

                                  linewidths=2., 

                                  use_bus_geodata=True)



lc = plot.create_line_collection(net, 

                                 lines=net.line.index, 

                                 color="grey",

                                 alpha=0.8, 

                                 linestyles="dashed", 

                                 linewidths=2.)



sc = plot.create_bus_collection(net, 

                                net.ext_grid.bus.values, 

                                patch_type="rect", 

                                size=200,

                                color="c", 

                                zorder=11)



plot.draw_collections([lc, lcd, bc, sc]);
long_lines = net.line[net.line.length_km > 2.].index

lc = plot.create_line_collection(net, net.line.index, color="grey", zorder=1)

lcl = plot.create_line_collection(net, long_lines, color="g", zorder=2)

pp.runpp(net)

low_voltage_buses = net.res_bus[net.res_bus.vm_pu < 0.98].index

bc = plot.create_bus_collection(net, net.bus.index, size=90, color="red", zorder=10)

bch = plot.create_bus_collection(net, low_voltage_buses, size=90, color="yellow", zorder=11)

plot.draw_collections([lc, lcl, bc, bch], figsize=(8,6));
net.bus.head()
net.bus.type.unique()