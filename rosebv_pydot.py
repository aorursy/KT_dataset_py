import pydot_ng as pydot
from hashlib import sha256
g = pydot.Dot()
g.set_type('digraph')
node = pydot.Node('legend')
node.set("shape", 'box')
g.add_node(node)
node.set('label', 'mine')