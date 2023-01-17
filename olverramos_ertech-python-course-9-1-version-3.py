class BinaryTree:
    _value = None
    _left = None
    _right = None
    
    def __init__(self, value):
        self._value = value
        
    def set_left(self, node):
        self._left = node
    
    def set_right(self, node):
        self._right = node
        
    def get_left(self):
        return self._left
    
    def get_right(self):
        return self._right
    
    def get_value(self):
        return self._value
     
    def set_value(self, value):
        self._value = value
    
def crear_arbol ():
    A9 = BinaryTree(9)
    A10 = BinaryTree(10)
    A14 = BinaryTree(14)
    A19 = BinaryTree(19)
    A20 = BinaryTree(20)
    A23 = BinaryTree(23)
    A25 = BinaryTree(25)
    A29 = BinaryTree(29)
    A27 = BinaryTree(27)
    A36 = BinaryTree(36)
    A39 = BinaryTree(39)

    root = A25
    root.set_left(A14)
    A25.set_right(A36)
    A14.set_left(A9)
    A14.set_right(A19)
    A36.set_left(A29)
    A36.set_right(A39)
    A9.set_right(A10)
    A19.set_right(A23)
    A29.set_left(A27)
    A23.set_left(A20)
    
    return root
# R/ 
class BinaryTree:
    _value = None
    _left = None
    _right = None
    
    def __init__(self, value):
        self._value = value
        
    def set_left(self, node):
        self._left = node
    
    def set_right(self, node):
        self._right = node
        
    def get_left(self):
        return self._left
    
    def get_right(self):
        return self._right
    
    def get_value(self):
        return self._value
     
    def set_value(self, value):
        self._value = value
    
    def preorder_print(self):
        if self._value is None:
            return
        
        print (self._value, end=" ")
        
        if self._left is not None:
            self._left.preorder_print()
        
        if self._right is not None:
            self._right.preorder_print()
root = crear_arbol()
root.preorder_print()
# R/
class BinaryTree:
    _value = None
    _left = None
    _right = None
    
    def __init__(self, value):
        self._value = value
        
    def set_left(self, node):
        self._left = node
    
    def set_right(self, node):
        self._right = node
        
    def get_left(self):
        return self._left
    
    def get_right(self):
        return self._right
    
    def get_value(self):
        return self._value
     
    def set_value(self, value):
        self._value = value
    
    def preorder_print(self):
        if self._value is None:
            return
        
        print (self._value, end=" ")
        
        if self._left is not None:
            self._left.preorder_print()
        
        if self._right is not None:
            self._right.preorder_print()
    
    def inorder_print(self):
        if self._value is None:
            return
        
        if self._left is not None:
            self._left.inorder_print()

        print (self._value, end=" ")
        
        if self._right is not None:
            self._right.inorder_print()
    
        
root = crear_arbol()
root.inorder_print()
# R/
class BinaryTree:
    _value = None
    _left = None
    _right = None
    
    def __init__(self, value):
        self._value = value
        
    def set_left(self, node):
        self._left = node
    
    def set_right(self, node):
        self._right = node
        
    def get_left(self):
        return self._left
    
    def get_right(self):
        return self._right
    
    def get_value(self):
        return self._value
     
    def set_value(self, value):
        self._value = value
    
    def preorder_print(self):
        if self._value is None:
            return
        
        print (self._value, end=" ")
        
        if self._left is not None:
            self._left.preorder_print()
        
        if self._right is not None:
            self._right.preorder_print()
    
    def inorder_print(self):
        if self._value is None:
            return
        
        if self._left is not None:
            self._left.inorder_print()

        print (self._value, end=" ")
        
        if self._right is not None:
            self._right.inorder_print()
    
    def postorder_print(self):
        if self._value is None:
            return
        
        if self._left is not None:
            self._left.postorder_print()
        
        if self._right is not None:
            self._right.postorder_print()
    
        print (self._value, end=" ")

root = crear_arbol()
root.postorder_print()
class BinaryTree:
    _value = None
    _left = None
    _right = None
    
    def __init__(self, value):
        self._value = value
        
    def set_left(self, node):
        self._left = node
    
    def set_right(self, node):
        self._right = node
        
    def get_left(self):
        return self._left
    
    def get_right(self):
        return self._right
    
    def get_value(self):
        return self._value
     
    def set_value(self, value):
        self._value = value
    
    def preorder(self, function):
        if self._value is None:
            return
        
        function (self._value)
        
        if self._left is not None:
            self._left.preorder(function)
        
        if self._right is not None:
            self._right.preorder(function)
    
    def inorder(self, function):
        if self._value is None:
            return
        
        if self._left is not None:
            self._left.inorder(function)

        function (self._value)
        
        if self._right is not None:
            self._right.inorder(function)
    
    def postorder(self, function):
        if self._value is None:
            return
        
        if self._left is not None:
            self._left.postorder(function)
        
        if self._right is not None:
            self._right.postorder(function)
    
        function (self._value)

root = crear_arbol()
def imprimir(x):
    print (x, end=" ")
    
print ('Post-Order')
root.postorder(imprimir)
print ('\nIn-Order')
root.inorder(imprimir)
print ('\nPre-Order')
root.preorder(imprimir)
root = crear_arbol()
def cuadrado(x):
    print (x**2, end=' ')
    
print ('Post-Order')
root.postorder(cuadrado)
print ('\nIn-Order')
root.inorder(cuadrado)
print ('\nPre-Order')
root.preorder(cuadrado)
# R/
class SearchBinaryTree:
    _value = None
    _left = None
    _right = None
    
    def __init__(self, value):
        self._value = value
   
    def insert(self, value):
        if value is None or self._value is None:
            return
        node = SearchBinaryTree(value)
        if value < self._value:
            if self._left is None:
                self._left = node
            else:
                self._left.insert(value)
        else:
            if self._right is None:
                self._right = node
            else:
                self._right.insert(value)
             
    def get_left(self):
        return self._left
    
    def get_right(self):
        return self._right
    
    def get_value(self):
        return self._value
     
    def set_value(self, value):
        self._value = value
    
    def preorder_print(self):
        if self._value is None:
            return
        
        print (self._value, end=" ")
        
        if self._left is not None:
            self._left.preorder_print()
        
        if self._right is not None:
            self._right.preorder_print()
    
    def inorder_print(self):
        if self._value is None:
            return
        
        if self._left is not None:
            self._left.inorder_print()

        print (self._value, end=" ")
        
        if self._right is not None:
            self._right.inorder_print()
    
        
    def postorder_print(self):
        if self._value is None:
            return
        
        if self._left is not None:
            self._left.postorder_print()
        
        if self._right is not None:
            self._right.postorder_print()
    
        print (self._value, end=" ")

def crear_arbol_ordenado ():
    root = SearchBinaryTree(25)
    root.insert(9)
    root.insert(10)
    root.insert(14)
    root.insert(19)
    root.insert(20)
    root.insert(23)
    root.insert(29)
    root.insert(27)
    root.insert(36)
    root.insert(39)
    
    return root

root = crear_arbol_ordenado()
root.insert(15)
#root.inorder_print()
root.postorder_print()
# R/
# R/
class SearchBinaryTree:
    _value = None
    _left = None
    _right = None
    
    def __init__(self, value):
        self._value = value
   
    def insert(self, value):
        if value is None or self._value is None:
            return
        node = SearchBinaryTree(value)
        if value < self._value:
            if self._left is None:
                self._left = node
            else:
                self._left.insert(value)
        else:
            if self._right is None:
                self._right = node
            else:
                self._right.insert(value)
             
    def get_left(self):
        return self._left
    
    def get_right(self):
        return self._right
    
    def get_value(self):
        return self._value
     
    def set_value(self, value):
        self._value = value
    
    def preorder_print(self):
        if self._value is None:
            return
        
        print (self._value, end=" ")
        
        if self._left is not None:
            self._left.preorder_print()
        
        if self._right is not None:
            self._right.preorder_print()
    
    def inorder_print(self):
        if self._value is None:
            return
        
        if self._left is not None:
            self._left.inorder_print()

        print (self._value, end=" ")
        
        if self._right is not None:
            self._right.inorder_print()
    
    def postorder_print(self):
        if self._value is None:
            return
        if self._left is not None:
            self._left.postorder_print()
        
        if self._right is not None:
            self._right.postorder_print()
    
        print (self._value, end=" ")

    def search(self, value):
        if self._value == value:
            return self
        if self._left is not None and value < self._value:
            return self._left.search(value)
        if self._right is not None and value > self._value:
            return self._right.search(value)
        
        
root = crear_arbol_ordenado()
root.insert(15)
#root.inorder_print()
b = root.search(25)
print(b)
print(b.get_value())
print(b.get_left().get_value())
# R/
    
class Node:
    _parent = None
    _value = None
    _left = None
    _right = None

    def __init__(self, value, parent=None):
        self._value = value
        self._parent = parent

    def insert(self, value):
        if value is None or self._value is None:
            return
        if value < self._value:
            if self._left is None:
                self._left = Node(value, self)
            else:
                self._left.insert(value)
        else:
            if self._right is None:
                self._right = Node(value, self)
            else:
                self._right.insert(value)

    def set_parent(self, parent):
        self._parent = parent

    def get_parent(self):
        return self._parent

    def get_left(self):
        return self._left

    def get_right(self):
        return self._right

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def set_left(self, node):
        self._left = node
        if node is not None:
            node.set_parent(self)

    def set_right(self, node):
        self._right = node
        if node is not None:
            node.set_parent(self)

    def preorder_print(self):
        if self._value is None:
            return

        print (self._value, end=" ")

        if self._left is not None:
            self._left.preorder_print()

        if self._right is not None:
            self._right.preorder_print()

    def inorder_print(self):
        if self._value is None:
            return

        if self._left is not None:
            self._left.inorder_print()

        print (self._value, end=" ")

        if self._right is not None:
            self._right.inorder_print()

    def postorder_print(self):
        if self._value is None:
            return

        if self._left is not None:
            self._left.postorder_print()

        if self._right is not None:
            self._right.postorder_print()

        print (self._value, end=" ")

    def search(self, value):
        if self._value == value:
            return self
        if self._left is not None and value < self._value:
            return self._left.search(value)
        if self._right is not None and value > self._value:
            return self._right.search(value)
        return None

    def is_leaf(self):
        return self._left is None and self._right is None
    
    def get_min(self):
        if self._left is None:
            return self
        return self._left.get_min()
    
    def get_max(self):
        if self._right is None:
            return self
        return self._right.get_max()
    
    def __lt__(self, node):
        return self._value < node.get_value()
    
    def __gt__(self, node):
        return self._value > node.get_value()
    
    def __del__(self):
        self._value = None
        self._left = None
        self._right = None
        self._parent = None
    
    def __repr__(self):
        result = "<<"
        if self._left is not None:
            result += f"{self._left.get_value()}"
        result += f",{self._value},"
        if self._right is not None:
            result += f"{self._right.get_value()}"
        result += ','
        if self._parent is not None:
            result += f"{self._parent.get_value()}"
        result += ">>"
        return result 
    
class SearchBinaryTree:
    _root = None
    
    def __init__(self):
        self._root = None
    
    def get_min(self):
        if self._root is not None:
            return self._root.get_min()

    def get_max(self):
        if self._root is not None:
            return self._root.get_max()
        
    def insert(self, value):
        if self._root is None:
            self._root = Node(value) 
        else:
            self._root.insert(value)
    
    def search(self, value):
        if self._root is not None:
            node = self._root.search(value)
            if node is not None:
                return True
        return False
    
    def print_tree(self, scheme='PRE'):
        if self._root is None:
            return
        if scheme.upper() == 'PRE':
            self._root.preorder_print()
        if scheme.upper() == 'IN':
            self._root.inorder_print()
        if scheme.upper() == 'POST':
            self._root.postorder_print()
        print ()
            
    def delete(self, value):
        if self._root is None:
            return False
        
        node_to_delete = self._root.search(value)
        if node_to_delete is None:
            return False
        
        node_to_delete_parent = node_to_delete.get_parent()
        
        if node_to_delete.is_leaf():
            replace_node = None
        elif node_to_delete._left is None:
            replace_node = node_to_delete._right.get_min()
        else:
            replace_node = node_to_delete._left.get_max()

        if replace_node is not None:
            replace_node_parent = replace_node.get_parent()
            if replace_node_parent < replace_node:
                replace_node_parent.set_right(None)
            else:
                replace_node_parent.set_left(None)
                
        if node_to_delete_parent is not None:
            if node_to_delete_parent > node_to_delete:
                node_to_delete_parent.set_left(replace_node)
            else:
                node_to_delete_parent.set_right(replace_node)
        else:
            self._root = replace_node
            
        if replace_node is not None:
            replace_node.set_left(node_to_delete.get_left())
            replace_node.set_right(node_to_delete.get_right())

        del node_to_delete
            

def crear_arbol_ordenado ():
    tree = SearchBinaryTree()
    
    for v in (25,14,36,9,19,29,39,10,15,22,27,32,20,24,31,34,17,21,23,16,18):
        tree.insert(v)
    
    return tree
tree = crear_arbol_ordenado()
print ('PREORDEN')
tree.print_tree('PRE')
print ('INORDEN')
tree.print_tree('IN')
print ('POSTORDEN')
tree.print_tree('POST')
print ('INORDEN')
tree.print_tree('IN')
tree.delete(19)
print ('INORDEN')
tree.print_tree('IN')
tree.print_tree('POST')
# R/
# R/
# R/