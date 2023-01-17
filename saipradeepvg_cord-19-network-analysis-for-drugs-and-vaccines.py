!pip install pyvis
from pyvis.network import Network

import networkx as nx

import pandas as pd

from IPython.display import IFrame
from pyvis.network import Network

import networkx as nx

import pandas as pd

import sys



VERSION_DIR='/kaggle/input/cord19/drug_submission_v2.3/'

CONETZ_NETWORK_FILE = VERSION_DIR+'conetz.tsv'

CONETZ_SP_NETWORK_FILE = VERSION_DIR+'conetz_specialized.tsv'

NETWORK_FILE = CONETZ_SP_NETWORK_FILE

ENTITY_NAME_FILE = VERSION_DIR+'resources/entity_name.tsv'

ENT_METADATA_FILE = VERSION_DIR+'resources/entity_metadata.csv'

ent_name=None

ent_id_name=None

ent_name_id={}

enttype_map=None

ent_cmap=None

ent_srcmap=None

connected_nodes=False

notebook_mode=True

overlap_ents=None

net_options = {

  "nodes": {

    "borderWidth": 0,

    "borderWidthSelected": 1,

    "scaling": {

      "min": 46

    }

  },

  "edges": {

    "color": {

      "inherit": True

    },

    "shadow": {

      "enabled": True

    },

    "smooth": True

  },

  "interaction": {

    "hover": True,

    "navigationButtons": True

  },

  "physics": {

    "enabled": True,

    "forceAtlas2Based": {

      "gravitationalConstant": -208,

      "springLength": 30

    },

    "minVelocity": 0.05,

    "timestep":0.1,

    "solver": "forceAtlas2Based"

  }

}
def getEntityMaps():

    ent_meta_map = pd.read_csv(ENT_METADATA_FILE, sep=',')

    enttype_map = ent_meta_map[['entid','enttype','entsource']]

    

    ent_name_df = pd.read_csv(ENTITY_NAME_FILE, sep='\t', converters={'TypeId':str})

    ent_name_df.TypeId=ent_name_df.TypeId.str.upper()

    ent_name_df.Synonym=ent_name_df.Synonym.str.upper()

    ent_name_df.DictId = ent_name_df.DictId.map(enttype_map.set_index('entid')['enttype'])

    

    ent_name = ent_name_df.set_index('Synonym').to_dict()

    overlap_ents = ent_name_df[ent_name_df.Synonym.isin(ent_name_df.Synonym[ent_name_df.Synonym.duplicated()])]

    ent_id_name = ent_name_df.set_index(['TypeId','DictId']).to_dict()

    

    ent_color = ent_meta_map[['enttype','entcolor']]

    ent_cmap = ent_color.set_index('enttype').to_dict(orient='index')

    

    ent_source = ent_meta_map[['enttype','entsource']]

    ent_srcmap = ent_source.set_index('enttype').to_dict(orient='index')

    return enttype_map, ent_name, ent_id_name, overlap_ents, ent_cmap, ent_srcmap



def getNetwork():

    nw = pd.read_csv(NETWORK_FILE,sep='\t',converters={'src_ent':str, 'target_ent':str})

    nw.src_type = nw.src_type.map(enttype_map.set_index('entid')['enttype'])

    nw.target_type = nw.target_type.map(enttype_map.set_index('entid')['enttype'])

    nw.src_ent=nw.src_ent.str.upper()

    nw.target_ent=nw.target_ent.str.upper()

    return nw



def buildQueryCriteria(src_ents, source_ent_types=None, target_ents=None, target_ent_types=None, 

                       queryByEntityName=True, topk=50, topkByType=None, connected_nodes=False, 

                       indirect_links=None, inference=False):

    # Normalize it upper-case

    src_ents = [i.strip().upper() for i in src_ents]

    

    criteria = {'src_ents' : src_ents,

                'src_ent_types': source_ent_types,

                'target_ents': target_ents,

                'target_ent_types': target_ent_types,

                'query_entname' : queryByEntityName,

                'topk' : topk,

                'topkByType' : topkByType,

                'connected_nodes' : connected_nodes,

                'indirect_links' : indirect_links,

                'inference':inference

                }

    return criteria



def queryByEntityTypes(nw, src_ent_types, target_ent_types):

    #Fetch the network

    qnw=None

    if(src_ent_types is not None):

        qnw = nw[nw.src_type.isin(src_ent_types)]

    if(target_ent_types is not None):

        qnw = nw[nw.target_type.isin(target_ent_types)]

    if(qnw is None):

        qnw=nw

    return qnw



def queryByEntityID(nw, src_ents, target_ents=None):

    #Fetch the network

    qnw = nw[nw.src_ent.isin(src_ents)]

    if(target_ents is not None):

        target_ents = [i.strip().upper() for i in target_ents]

        qnw = nw[nw.target_ent.isin(target_ents)]

    return qnw



def getEntityIds(ents):

    # Normalize it upper-case

    for i in ents:

        if(i not in ent_name['TypeId']):

            print('Term Error : Oops! Query term ['+i+'] NOT Found ')

            print('Suggestion : Please use Type ID as query from the source DB stated in the metadata OR remove the term')

            sys.exit(0)

        ov = overlap_ents[overlap_ents.Synonym==i]

        if len(ov)>1:

            print('Entity Conflict Error : Oops! Found more than one entity type for the query term '+i)

            print('Suggestion : Please use one of the following Type IDs as query to resolve the conflict \n\tand search by setting QueryByName=False : ')

            for idx in ov.index:

                print('\tFor '+ov.DictId[idx]+' Dictionary; use : QueryTerms=[\''+ov.TypeId[idx]+'\']')

            sys.exit(0)

    

    print(' Querying by Entity Name ..')

        

    #Get the entity triplet for the entities

    typeids = [ent_name['TypeId'][i] for i in ents]

    dictids = [ent_name['DictId'][i] for i in ents]

    #Update Entity Name to Entity/Node ID for reference

    for i in range(len(ents)):

        ent_name_id[ents[i]]=getNodeID(typeids[i], dictids[i])

    return typeids, dictids





def queryByEntityName(nw, src_ents, target_ents=None):

    typeids, dictids = getEntityIds(src_ents)

    qnw = nw[nw.src_ent.isin(typeids) & (nw.src_type.isin(dictids))]

    if(target_ents is not None):

        target_ents = [i.upper() for i in target_ents]

        typeids, dictids = getEntityIds(target_ents)

        

        qnw = nw[(nw.src_ent.isin(typeids)) & (nw.src_type.isin(dictids))]

    return qnw



def queryTopk(nw, topk, topkByType):

    qnw=None

    if(topkByType!=None):

        qnw = nw.groupby(['src_ent','target_type']).head(topkByType)

    else:

        qnw = nw.groupby(['src_ent','target_type']).head(topk)

    return qnw



def queryInferredEdges(nw):

    qnw = nw[nw.debug=='I']

    return qnw



def queryNetwork(nw, criteria):

    # Use the criteria to query the network by entity name

    qnw=None

    

    if(criteria['query_entname']==True):

        qnw = queryByEntityName(nw, criteria['src_ents'], target_ents=criteria['target_ents'])

    else:

        qnw = queryByEntityID(nw, criteria['src_ents'], target_ents=criteria['target_ents'])

    

    # Display only inferred edges

    if(criteria['inference']==True):

        qnw = queryInferredEdges(qnw)

    else:

        qnw = qnw[qnw.debug!='I']

    

    # Query by entity types

    qnw = queryByEntityTypes(qnw, criteria['src_ent_types'], criteria['target_ent_types'])

    

    # Display only Top-k entites

    qnw = queryTopk(qnw, criteria['topk'], criteria['topkByType'])

    

    return qnw



def getEntityNames(src, target):

    src_name = ent_id_name['Synonym'][src]

    target_name = ent_id_name['Synonym'][target]

    return src_name, target_name



def getNodeID(typeid, dictid):

    return typeid+'-'+dictid[:2]



def buildNodeAttributes(e):

    # Build Node attributes - node_id, node_label, node_title, node_color 

    src_label, target_label = getEntityNames((e[0],e[1]), (e[2],e[3]))

    

    # Build src node

    src_id = getNodeID(e[0], e[1])

    src_title="<b>"+src_label+"</b><br><i>"+e[1]+"<br>"+e[0]+"</i><br>"+ent_srcmap[e[1]]['entsource']

    src_color=ent_cmap[e[1]]['entcolor']

    

    # Build target node

    target_id = getNodeID(e[2], e[3])

    target_title="<b>"+target_label+"</b><br><i>"+e[3]+"<br>"+e[2]+"</i><br>"+ent_srcmap[e[3]]['entsource']

    target_color=ent_cmap[e[3]]['entcolor']

    

    return (src_id, src_label, src_title, src_color), (target_id, target_label, target_title, target_color)



def edgeAttributes(ent1, ent2, edge_props):

    #Build edge attributes

    edge_title = '<b>'+ent1+' --- '+ent2+'</b><br>Article Evidence(s) :<br>'

    

    if('I' in edge_props):

        num_arts=0

        edge_title+='<b>Inferred from GCAS </b></i>'

    else:

        edge_prop_arr = edge_props.split(sep=',')

        num_arts = int(edge_prop_arr[0])-3

        art_type=''

        for i in range(3, len(edge_prop_arr)):

            art=edge_prop_arr[i].replace("[","")

            art=art.replace("]","")

            if("FT_" in art):

                art=art.replace("FT_","")

                art_type='CORD_UID :'

            else:

                art_type='PUBMED_ID :'

            edge_title+=art_type+'<i>'+art+'</i><br>'

        if(num_arts>5):

            edge_title+='and <i><b>'+str(num_arts)+'</b> more articles ...</i>'

    

    return edge_title



def buildGraph(G, criteria, filters=False):

    #Define Network layout

    net = Network(height="1024px", width="100%", bgcolor="white", font_color="black", notebook=notebook_mode)

    net.options=net_options

    

    #Convert networkx G to pyvis network

    edges = G.edges(data=True)

    nodes = G.nodes(data=True)

    if len(edges) > 0:

        for e in edges:

            snode_attr=nodes[e[0]]

            tnode_attr=nodes[e[1]]            

            net.add_node(e[0], snode_attr['label'], title=snode_attr['title'], color=snode_attr['color'])

            net.add_node(e[1], tnode_attr['label'], title=tnode_attr['title'], color=tnode_attr['color'])

            if(criteria['inference']==True):

                net.add_edge(e[0], e[1], width=2, title=e[2]['title'])

            else:

                net.add_edge(e[0], e[1], value=e[2]['value'], title=e[2]['title'])

    return net    



def applyGraphFilters(G, criteria):

    

    fnodes={}

    # Filter1 - Connected nodes

    if(criteria['connected_nodes']):    

        bic = nx.biconnected_components(G)

        for i in bic:

            if(len(i)>2):

                fnodes=i.union(fnodes)

    

        # Get the sub-graph after applying the filter(s)

        G=G.subgraph(fnodes)

    

    # Filter2 - 'indirect_links'

    il_dicts = criteria['indirect_links']

    if(il_dicts is not None):    

        snode = il_dicts['source_node'] if ('source_node' in il_dicts) else criteria['src_ents'][0]

        snode = ent_name_id[snode.upper()]

        #Depth=Hops+1

        depth=(il_dicts['hops']+1) if('hops' in il_dicts) else 2

        

        if('target_nodes' in il_dicts):

            tnodes = il_dicts['target_nodes']

        elif(criteria['target_ents'] is not None):

            tnodes = criteria['target_ents']

        else:

            tnodes=criteria['src_ents']        

        tnodes = [ ent_name_id[i.upper()] for i in  tnodes]

    

        # Traverse k-hops from source to target nodes.            

        paths_between_generator = nx.all_simple_paths(G, source=snode, target=tnodes, cutoff=depth)

        indirect_paths=[]

        i=0

        for k, path in enumerate(paths_between_generator):

            #if(len(path)==depth+1):

            ce=[]

            #print(path)

            for j, e in enumerate(path):

                if j+1 <= len(path)-1:

                    ce.append((path[j], path[j+1]))

            indirect_paths.extend(ce)

        G=G.edge_subgraph(indirect_paths)

    return G



def run(criteria):

    

    print('Building CoNetz from '+NETWORK_FILE)

    

    # Load the entire network

    nw_df = getNetwork()



    # Query the network with the defined search criteria

    qnw = queryNetwork(nw_df, criteria)

    

    # Build association network using the query result

    sources = qnw['src_ent']

    source_types=qnw['src_type']

    targets = qnw['target_ent']

    target_types=qnw['target_type']

    weights = qnw['score']

    stats = qnw['debug']

    edge_data = zip(sources, source_types, targets, target_types, weights, stats)



    G=nx.Graph()

    for e in edge_data:

        snode, tnode = buildNodeAttributes(e)

        G.add_node(snode[0], label=snode[1], title=snode[2], color=snode[3])

        G.add_node(tnode[0], label=tnode[1], title=tnode[2], color=tnode[3])

        G.add_edge(snode[0], tnode[0], value=e[4], title=edgeAttributes(snode[1],tnode[1], e[5]))



    applyFilter = (criteria['connected_nodes'] or criteria['indirect_links'])

    if(applyFilter):

        G=applyGraphFilters(G, criteria)



    net = buildGraph(G, criteria, filters=applyFilter)

    if(criteria['inference']==True):

        net.options['edges']['dashes']=True

    else:

        net.options['edges']['dashes']=False



    nassocs = len(G.edges())

    print(' Number of Associations in the Network -->'+str(nassocs))

    if(nassocs==0):

        print(' No Associations found for the query, Please change your Query ')

        sys.exit(0)

    

    return net
# Prepare Entity Maps

enttype_map, ent_name, ent_id_name, overlap_ents, ent_cmap, ent_srcmap = getEntityMaps()

#Display Entity Types

enttype_map
QueryTerms=['C000657245']

criteria = buildQueryCriteria(QueryTerms, topkByType=15, queryByEntityName=False)

criteria

net = run(criteria)

net.show("cord19_ex1.html")
QueryTerms=['C000657245']

criteria = buildQueryCriteria(QueryTerms,target_ent_types=['DRUGS','CHEMICALS'], topkByType=15, queryByEntityName=False)

criteria

net = run(criteria)

net.show("cord19_ex2.html")
QueryTerms=['naproxen', 'clarithromycin', 'minocycline', 'covid-19']

criteria = buildQueryCriteria(QueryTerms, topkByType=10)

criteria

net = run(criteria)

net.show("cord19_ex3.html")
QueryTerms=['naproxen', 'clarithromycin', 'minocycline', 'covid-19']

criteria = buildQueryCriteria(QueryTerms, topkByType=10, connected_nodes=True)

criteria

net = run(criteria)

net.show("cord19_ex4.html")
QueryTerms=['covid-19']

criteria = buildQueryCriteria(QueryTerms,target_ent_types=['DISEASE','DRUGS'], topkByType=10)

print(criteria)

net = run(criteria)

net.show("cord19_ex5.html")
QueryTerms=['baricitinib','covid-19', 'ace2']

criteria = buildQueryCriteria(QueryTerms, topkByType=15, target_ent_types=['HGNC','DRUGS','GENE_SARS-CoV-2','ORGANISM', 'CHEMICALS', 'VIROLOGY TERMS', 'DISEASE'],

                              indirect_links={'source_node':'covid-19', 'target_nodes':['baricitinib'], 'hops':3})

criteria

net = run(criteria)

net.show("cord19_ex6.html")
QueryTerms=['C000657245','D017963', 'NS8_WCPV', 'GO:0019079', 'HGNC:14512']

criteria = buildQueryCriteria(QueryTerms, topkByType=20, queryByEntityName=False, connected_nodes=True)

criteria

net = run(criteria)

net.show("cord19_ex7.html")
QueryTerms=['C000657245']

criteria = buildQueryCriteria(QueryTerms, topkByType=40, queryByEntityName=False, inference=True)

criteria

net = run(criteria)

net.show("cord19_inference.html")