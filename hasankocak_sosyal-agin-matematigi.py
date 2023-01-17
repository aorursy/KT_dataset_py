import networkx as nx
G_symmetric = nx.Graph()

G_symmetric.add_edge('Amitabh Bachchan','Abhishek Bachchan')

G_symmetric.add_edge('Amitabh Bachchan','Aamir Khan')

G_symmetric.add_edge('Amitabh Bachchan','Akshay Kumar')

G_symmetric.add_edge('Amitabh Bachchan','Dev Anand')

G_symmetric.add_edge('Abhishek Bachchan','Aamir Khan')

G_symmetric.add_edge('Abhishek Bachchan','Akshay Kumar')

G_symmetric.add_edge('Abhishek Bachchan','Dev Anand')

G_symmetric.add_edge('Dev Anand','Aamir Khan')

nx.draw_networkx(G_symmetric)
G_asymmetric = nx.DiGraph()

G_asymmetric.add_edge('A','B')

G_asymmetric.add_edge('A','D')

G_asymmetric.add_edge('C','A')

G_asymmetric.add_edge('D','E')

nx.spring_layout(G_asymmetric)

nx.draw_networkx(G_asymmetric)
G_weighted = nx.Graph()

G_weighted.add_edge('Amitabh Bachchan','Abhishek Bachchan', weight=25)

G_weighted.add_edge('Amitabh Bachchan','Aaamir Khan', weight=8)

G_weighted.add_edge('Amitabh Bachchan','Akshay Kumar', weight=11)

G_weighted.add_edge('Amitabh Bachchan','Dev Anand', weight=1)

G_weighted.add_edge('Abhishek Bachchan','Aaamir Khan', weight=4)

G_weighted.add_edge('Abhishek Bachchan','Akshay Kumar',weight=7)

G_weighted.add_edge('Abhishek Bachchan','Dev Anand', weight=1)

G_weighted.add_edge('Dev Anand','Aaamir Khan',weight=1)

nx.spring_layout(G_weighted)

nx.draw_networkx(G_weighted)