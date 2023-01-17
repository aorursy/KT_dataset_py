from glob import glob

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from ortools.graph import pywrapgraph
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def list_lines(file_name):
    """Returns a list of integer lists."""
    with open(file_name) as file:
        lines = file.read().splitlines()
    line_list = [[int(n) for  n in ll.split()] for ll in lines]
    return line_list


def set_params(line_list):
    top_line = line_list[0]
    params = {'DRONE_COUNT': top_line[2],
              'WT_CAP': top_line[4],
              'END_TIME': top_line[3],
              }
    return params


def find_wh_lines(line_list):
    """Provides the dividing line between warehouse and
    order sections in the line list."""
    wh_count = line_list[3][0]
    wh_endline = (wh_count*2)+4
    return wh_endline


def get_weights(line_list):
    weights = np.array(line_list[2])
    return weights.astype(np.int16)


def get_inventories(line_list):
    """Returns a 2-d array of P products by W warehouses."""
    wh_endline = find_wh_lines(line_list)
    invs = line_list[5:wh_endline+1:2]
    supply = np.array(invs).transpose()
    return supply.astype(np.int16)


def get_orders(line_list):
    """Returns a 2-d array of P products by C orders."""
    wh_endline = find_wh_lines(line_list)
    demand = np.zeros((line_list[1][0], line_list[wh_endline][0]),
                            dtype=np.int16)
    orders = line_list[wh_endline+3::3]
    for i,ord in enumerate(orders):
        for prod in ord:
            demand[prod, i] += 1
    return demand.astype(np.int16)


def get_locs(line_list):
    wh_endline = find_wh_lines(line_list)
    wh_locs = np.array(line_list[4:wh_endline:2])
    cust_locs = np.array(line_list[wh_endline+1::3])
    return wh_locs.astype(np.int16), cust_locs.astype(np.int16)
# main
files = ['../input/hashcode-drone-delivery/busy_day.in']
line_list = list_lines(files[0])

params = set_params(line_list)
supply = get_inventories(line_list)
demand = get_orders(line_list)
wh_locs, cust_locs = get_locs(line_list)
weights = get_weights(line_list)
import holoviews as hv
hv.extension('bokeh')
print(params)

freqs, edges = np.histogram(weights, 20)
wt_prod = hv.Histogram((edges, freqs)).options(xlabel="product weights"
                                               , width=250)

order_weights = (weights.reshape(weights.size, -1)* demand) \
                    .sum(axis=0)
freqs, edges = np.histogram(order_weights, 20)
wt_orders = hv.Histogram((edges, freqs)).options(xlabel="order weights",
                                                 width=400)

surplus = hv.Curve(supply.sum(axis=1) - demand.sum(axis=1)).options(width=500,
                                            xlabel='product', ylabel='surplus')

customers = hv.Points(np.fliplr(cust_locs)).options(width=600, height=400)
warehouses = hv.Points(np.fliplr(wh_locs)).options(size=8, alpha=0.5)

display(hv.Layout(wt_prod+wt_orders).options(shared_axes=False), surplus, 
            customers*warehouses)

def assign_whs(supply, wh_locs, demand, cust_locs):
    """OR-tools function to assign warehouses to orders using a max-flow min-cost
    solver. Numbering scheme is as follows:
        warehouses = 1250 to 1259
        customers/orders = 0 to 1249
    
    Supply and demand do not have to be equal.
    """
    assignments = []
    count = 0
    distances = distance_matrix(cust_locs, wh_locs)

    for i in range(400):  # iterate over products
        item_count = 0

        # Network description
        start_nodes = np.repeat(np.arange(1250,1260), 1250).tolist()
        end_nodes = np.tile(np.arange(0,1250), 10).tolist()        
        capacities = np.tile(demand[i], 10).tolist()
        costs = np.transpose(distances).ravel().astype(int).tolist()
        supplies = np.negative(demand[i]).tolist() + supply[i].tolist()
                                            # nodes in numerical order
        # Build solver
        min_cost_flow = pywrapgraph.SimpleMinCostFlow()

        for s in range(len(start_nodes)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(
                start_nodes[s], end_nodes[s], capacities[s], costs[s]
                )
        for s in range(len(supplies)):
            min_cost_flow.SetNodeSupply(s, supplies[s])

        # Solve
        if min_cost_flow.SolveMaxFlowWithMinCost() == min_cost_flow.OPTIMAL:
            for arc in range(min_cost_flow.NumArcs()):
                if min_cost_flow.Flow(arc) > 0:
                    warehouse = min_cost_flow.Tail(arc) - 1250
                    customer = min_cost_flow.Head(arc)
                    product = i
                    quant = min_cost_flow.Flow(arc)
                    cost = min_cost_flow.UnitCost(arc)
                    assign = [warehouse, customer, product, quant, cost]
                    assignments.append(assign)
                    item_count += quant
        count += item_count
    
    print(supply.sum(), demand.sum(), count)              
    return np.array(assignments)
# main
assignments = assign_whs(supply, wh_locs, demand, cust_locs)
assign_df = pd.DataFrame(assignments, columns=['wh', 'cust', 'prod_',
                                               'quant', 'dist'])
assign_df
def order_orders(df):

    customers = df.cust.unique()
    demand = df.groupby('cust')['quant'].sum()

    locs = np.vstack((cust_locs[customers], wh_locs[0]))

    distances = np.ceil(distance_matrix(locs, locs)).astype(int)

    customer_map = dict(zip(customers, range(len(customers))))

    data = {}
    data['dists'] = distances.tolist()
    data['drone_count'] = 1
    data['warehouse'] = len(locs) - 1

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data['dists']),
                                           data['drone_count'], data['warehouse'])

    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['dists'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Get vehicle routes
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
      index = solution.Value(routing.NextVar(index))
      route.append(manager.IndexToNode(index))
    routes.append(route[1:-1])

    # Single vehicle approximation
    route = routes[0]

    reverse_dict = {v: k for k,v in customer_map.items()}
    cust_ids = [reverse_dict[r] for r in route]

    df['cust_sort'] = pd.Categorical(df.cust, cust_ids)
    df = df.sort_values('cust_sort')
    return df
def load_drones(df):
    test_wt = 0
    load_wts = []
    df = df.sort_values('cust')
    for i,tup in enumerate(df.itertuples()):
        test_wt += tup.weight
        if test_wt <= params['WT_CAP']:
            load_wt = test_wt
        else:
            load_wt = tup.weight
            test_wt = tup.weight
        load_wts.append(load_wt)

    df['load_weight'] = load_wts
    df['load_tag'] = df.load_weight.eq(df.weight).cumsum()-1
    return df


def set_loads(assignments):
    assign_df = pd.DataFrame(assignments, columns=['wh', 'cust', 'prod_',
                                                   'quant', 'dist'])
    # Monster method chain to deal with quantities > 1 and define loads
    assign_df = assign_df.reindex(assign_df.index.repeat(assign_df.quant)) \
                         .reset_index(drop=True) \
                         .assign(quant=1,
                                 weight = lambda x: weights[x.prod_.to_numpy()],
                                 work = lambda x: x.dist * x.weight) \
                         .groupby('wh', as_index=False).apply(load_drones) \
                         .sort_values(['wh', 'cust', 'load_tag']) \
                         .reset_index(drop=True)
    return assign_df


def assign_drones(assign_df):
    wh_work = assign_df.groupby('wh')['work'].sum()
    drones_per_wh = (wh_work/ wh_work.sum()
                         * params['DRONE_COUNT'])
    drone_counts = drones_per_wh.round(0).astype(int)

    if drone_counts.sum() != params['DRONE_COUNT']:
        drone_counts = np.ediff1d(drones_per_wh.cumsum().round(0).astype(int),
                                        to_begin=drone_counts[0])

    drone_whs = np.repeat(np.arange(len(wh_locs)), drone_counts)
    drone_dict = dict(zip(range(params['DRONE_COUNT']), drone_whs))

    drone_assigns = {}
    for k, v in drone_dict.items():
        drone_assigns[v] = drone_assigns.get(v, []) + [k]

    df_list = []
    for grp, df in assign_df.groupby('wh'):
        drone_ids = drone_assigns[df.wh.iloc[0]]
        df['drone_id'] = df.load_tag % len(drone_ids) + min(drone_ids)
        df_list.append(df)

    df_end = pd.concat(df_list)
    return df_end


# main 
assign_df = set_loads(assignments)
df_end = assign_drones(assign_df)
df_end = df_end.groupby(['wh', 'cust', 'load_tag', 'drone_id', 'prod_'],
                            as_index=False)['quant'].sum()
df_end
def load_drones_improved(df):
    test_wt = 0
    load_wts = []
    for i,tup in enumerate(df.itertuples()):
        test_wt += tup.weight
        if test_wt <= params['WT_CAP']:
            load_wt = test_wt
        else:
            load_wt = tup.weight
            test_wt = tup.weight
        load_wts.append(load_wt)

    df['load_weight'] = load_wts
    df['load_tag'] = df.load_weight.eq(df.weight).cumsum()-1
    return df


def reset_loads(df_end_reordered):
    df_end_reordered = df_end_reordered.reset_index(drop=True) \
                                 .assign(weight = lambda x: weights[x.prod_.to_numpy()] * x.quant) \
                                 .groupby('wh', as_index=False).apply(load_drones_improved)
    return df_end_reordered


def assign_drones(df_end_reordered):
    df_list = []
    for grp, df in df_end_reordered.groupby('wh'):
        drone_ids = df.wh.iloc[0]
        df['drone_id'] = df.load_tag % df.drone_id.nunique() + min(df.drone_id)
        df_list.append(df)

    df_end = pd.concat(df_list)
    return df_end


df_end_reordered = df_end.groupby('wh').apply(order_orders) \
                         .pipe(reset_loads) \
                         .pipe(assign_drones)

df_end_reordered 
df_end_loading = df_end_reordered.groupby(['wh', 'load_tag', 'prod_'], as_index=False).agg(
                                    drone_id = ('drone_id', 'first'),
                                    quant = ('quant', 'sum'),
                                     weight_sum = ('weight', 'sum')
                                    )
df_end_loading
def write_instrux(df_load, df_deliver, sub):
    line_count_load = 0
    for tup in df_load.itertuples():
        sub.write(f"{tup.drone_id} L {tup.wh} {tup.prod_} {tup.quant}\n")
        line_count_load +=1
    for tup in df_deliver.itertuples():
        sub.write(f"{tup.drone_id} D {tup.cust} {tup.prod_} {tup.quant}\n")
        line_count_load +=1
    return line_count_load


with open('submission.csv', 'w') as sub:
    sub.write(f"{len(df_end_loading) + len(df_end_reordered)}\n")
    line_count = 0    
    
    drone_tag = df_end_loading.drop_duplicates(['drone_id', 'load_tag'])

    for dt in drone_tag.itertuples():
        df_load_tag = df_end_loading[(df_end_loading.load_tag == dt.load_tag) & \
                                          (df_end_loading.drone_id == dt.drone_id)]
        df_deliver_tag = df_end_reordered[(df_end_reordered.load_tag == dt.load_tag) & \
                                          (df_end_reordered.drone_id == dt.drone_id)]
    
        line_count_load = write_instrux(df_load_tag, df_deliver_tag, sub)
        line_count += line_count_load

print(len(df_end_loading) + len(df_end_reordered), line_count)
