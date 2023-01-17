import matplotlib.pyplot as plt
import networkx as nx


#################################################################
# Function name: incremental_graph
#
# Obtains the incremental graph
#################################################################
def incremental_graph(graph):
    
    inc_graph=nx.MultiDiGraph()
    inc_graph.add_nodes_from(graph.nodes)
    
    for (i,j) in graph.edges:
        
        b=graph[i][j]['b']      #lower bound of flow
        c=graph[i][j]['c']      #upper bound of flow
        f=graph[i][j]['f']      #flow
        
        if(f < c):
            inc_graph.add_edge(i,j,rc=(c-f),a='+')
        
        if(b < f):
            inc_graph.add_edge(j,i,rc=(f-b),a='-')
    
    return inc_graph




#################################################################
# Function name: ford_fulkerson
#
# Calculates the maximum flow of a graph with a compatible flow
#################################################################
def ford_fulkerson(graph, a0, source, sink):
    
    aux_graph=graph
    
    while True:
        
        inc_graph=incremental_graph(aux_graph)      #calculates the incremental graph
        
        nx.draw(inc_graph,pos,node_size=500,with_labels=True)  #and draws it
        plt.show()
        
        
        print('\nIncremental Graph\'s arcs:')
        min_rc=float("inf")
        
        #prints all the incremental graph's arcs
        for (i,j,k) in inc_graph.edges:                  
            e=inc_graph[i][j][0]['rc']
            print(i+j,' : ',inc_graph[i][j][0])
            
            #and calculates the lowest residual capacity (epsilon)
            if (e < min_rc):
                min_rc=e
        
        
        print('\nPaths:') 
        paths=[]
        #obtains all paths from source to sink
        for p in nx.all_simple_paths(inc_graph, source, sink):      
            paths.append(p)
        print(paths)
        
        
        print('\nFlow(a0):')
        print(a0[2]['f'])
        
        #breaks the loop when there are no more paths from source to sink
        if paths==[]:       
            break
        
        
        for i in range(1, len(paths[0])):    #for each arc in the path
            node1=paths[0][i-1]
            node2=paths[0][i]
            
            #adds epsilon to every arc a+ in the path
            if(inc_graph[node1][node2][0]['a']=='+'):   
                aux_graph[node1][node2]['f']+=min_rc
                
            #substracts epsilon to every arc a- in the path 
            if(inc_graph[node1][node2][0]['a']=='-'):   
                aux_graph[node1][node2]['f']-=min_rc        
        
        
        a0[2]['f']+=min_rc      #adds epsilon to the flow of a0        
        
        
        print('\n----------------------------------------------------------------------------')
    
    return aux_graph
    

graph = nx.DiGraph()

graph.add_nodes_from('ABCD')

graph.add_edges_from([
    ('A', 'B', {'b': 0, 'c': 3, 'f': 1}),
    ('A', 'C', {'b': 1, 'c': 4, 'f': 1}),
    ('A', 'D', {'b': 0, 'c': 2, 'f': 0}),
    ('B', 'D', {'b': 1, 'c': 4, 'f': 1}),
    ('C', 'D', {'b': 0, 'c': 5, 'f': 1})
    
    ])


source='A'
sink='D'

a0=(sink, source, {'b': 1, 'c': 9, 'f': 2})
pos = nx.circular_layout(graph)
nx.draw(graph,pos,node_size=500,with_labels=True)
plt.show()

print('Graph arcs:\n')
for (i,j) in graph.edges:
            print(i+j,' : ',graph[i][j])
        
print('\nSource:',source,'\tSink:',sink)

print('\na0 = ',a0[0]+a0[1],' : ',a0[2])
J = ford_fulkerson(graph, a0, source, sink)
print('Graph with the maximum flow:')
#J.add_edges_from([a0])
nx.draw(J,pos,node_size=500,with_labels=True)
plt.show()

print('Graph arcs:\n')

for (i,j) in J.edges:
    print(i+j,' : ',J[i][j])

print('\nSource:',source,'\tSink:',sink)

print('\na0 = ',a0[0]+a0[1],' : ',a0[2])