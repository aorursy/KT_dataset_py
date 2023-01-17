import spacy

from ipywidgets import interact, interactive, fixed, interact_manual





import plotly.graph_objects as go



import networkx as nx



import plotly.io as pio

pio.renderers.default = "kaggle"
pio.renderers





# In[3]:





import warnings



warnings.filterwarnings('ignore')





# In[ ]:











# In[4]:





from spacy.tokens import Token

Token.set_extension("show", default = True)





# In[5]:





def bianli_G(doc):

    root_node = find_root(doc)

    return bianli_node(root_node)

    

def bianli_node(node, show_all=False):

    yield node

    for c in node.children:

        if next(c.ancestors)._.show or show_all:

            for x in bianli_node(c):

                yield x

                

def bianli_node_stable(node):

    "直接返回列表"

    tmp = list(bianli_node(node, show_all=True))

    tmp.sort(key=lambda x : x.i)

    return tmp





# In[6]:





def find_root(doc):

    for token in doc:

        if token.dep_ == 'ROOT':

            return token



def make_G(doc):

    G = nx.DiGraph()

    for token in doc:

        # print(token.text, token.pos_, token.dep_, list(a.i for a in token.children))

        G.add_node(token)

        for a in token.children:

            G.add_edge(token, a)



    pos = nx.planar_layout(G)

    pos2 = {x:(b, a) for x, (a, b) in pos.items()}

    return G, pos2





# In[ ]:











# In[7]:





def _my_bianli_node(node, out, ceng=0):

    out[node.i] = ceng

    for c in node.children:

        _my_bianli_node(c, out, ceng+1)

        

def my_new_make_G(doc, width=800, height=400):

    G = nx.DiGraph()

    for token in doc:

        # print(token.text, token.pos_, token.dep_, list(a.i for a in token.children))

        G.add_node(token)

        for a in token.children:

            G.add_edge(token, a)

    

    # 我自己加入的部分

    ceng_of_nodes = [0] * len(G)

    root_node = find_root(doc)

    _my_bianli_node(root_node, ceng_of_nodes)

    ceng_max = max(ceng_of_nodes)

    ceng_of_nodes = [ceng_max-c for c in ceng_of_nodes]

    Ys = [c/ceng_max*height for c in ceng_of_nodes]

    Xs = [i/len(doc)*width for i in range(len(doc))]

    pos = {token: (Xs[token.i], Ys[token.i]) for token in doc}



    return G, pos





# In[8]:





def make_annotations(pos, doc, font_size=15, font_color=None):

    doc_show = list(bianli_G(doc))

    annotations = []

    for token in doc_show:

        text = token.text if token._.show else " ".join(a.text for a in bianli_node_stable(token))

        annotations.append(

            dict(

                text=text, # or replace labels with a different list for the text within the circle

                textangle = textangle,

                x=pos[token][0], y=pos[token][1],

                xref='x1', yref='y1',

                font=dict(color=font_color, size=font_size),

                showarrow=False)

        )

    return annotations



def get_plot_data(G, pos, doc):

    doc_show = list(bianli_G(doc))

    Xn=[pos[token][0] for token in doc_show]

    Yn=[pos[token][1] for token in doc_show]

    Xe=[]

    Ye=[]

    for edge_a, edge_b in G.edges:

        if edge_a in doc_show and edge_b in doc_show:

            Xe+=[pos[edge_a][0],pos[edge_b][0], None]

            Ye+=[pos[edge_a][1],pos[edge_b][1], None]



    text = [token.pos_ for token in doc_show]

    return Xn, Yn, Xe, Ye, text



def update_point(trace, points, selector):

    global f, G, doc, pos

    doc_show = list(bianli_G(doc))

    for i in points.point_inds:

        doc_show[i]._.show = not doc_show[i]._.show

        # print("Clicked", doc[i]._.show)

    f.update_layout(annotations=make_annotations(pos, doc), width=width, height=height)

    f.update_annotations(dict(

        xref="x",

        yref="y",

        showarrow=True,

        arrowhead=7,

        arrowcolor = arrowcolor,

        ax=arraw_x,

        ay=arraw_len

    ))

    Xn, Yn, Xe, Ye, text = get_plot_data(G, pos, doc)   

    f.data[0].x = Xe

    f.data[0].y = Ye

    f.data[1].x = Xn

    f.data[1].y = Yn

    

def plot_G(G, pos, doc):

    global f

    Xn, Yn, Xe, Ye, text =  get_plot_data(G, pos, doc)   

    f = go.FigureWidget([go.Scatter(x=Xe,

                   y=Ye,

                   mode='lines',

                   name='dep',

                   line=dict(color=line_color, width=1),

                   hoverinfo='none'

                   ),

                    go.Scatter(x=Xn,

                  y=Yn,

                  mode='markers',

                  name='tokens',

                  marker=dict(symbol='circle-dot',

                                size=18,

                                color=circle_fill_color,    

                                line=dict(color=circle_stroke_color, width=1)

                                ),

                  text=text,  # 之后可以考虑把这里做成中文词性说明

                  hoverinfo='text',

                  opacity=circle_opacity)

                    ])



    

    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title

            zeroline=False,

            showgrid=False,

            showticklabels=False,

            )

    

    



    

    f.update_layout(title= 'Spacy Visualizer V1',

            width=width,

            height=height,

            annotations=make_annotations(pos, doc, font_color = font_color),

            font_size=16,

            showlegend=False,

            xaxis=axis,

            yaxis=axis,

            margin=dict(l=100, r=100, b=85, t=100),

            hovermode='closest',

            plot_bgcolor=background_color

            )

    

    f.update_annotations(dict(

        xref="x",

        yref="y",

        showarrow=True,

        arrowhead=7,

        arrowcolor = arrowcolor,

        ax=arraw_x,

        ay=arraw_len

    ))

    

    scatter = f.data[1]

    scatter.on_click(update_point)



    return f
# 画图时候使用的常量

line_color = "#aa98d5"

circle_fill_color = "white"

circle_stroke_color = "white"

background_color = "white"

text_color = "black"

font_color = "black"

circle_opacity = 0.1

arraw_x = 0

arraw_len = 0

arrowcolor = 'green'

width = 1000

height = 800

textangle = 20



sentens = """Stone does not decay, and so the tools of long ago have remained when even the bones of the men who made them have disappeared without trace."""

def my_main(sentens=sentens):

    global f, G, doc, pos

    nlp = spacy.load("en_core_web_lg")

    doc = nlp(sentens)

    G, pos = my_new_make_G(doc)

    

    root_node = find_root(doc) # 根节点

    f = plot_G(G, pos, doc)

    return f
interact(my_main, sentens=sentens);