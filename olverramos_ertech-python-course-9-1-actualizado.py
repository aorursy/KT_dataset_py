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
# R/
# R/
# R/
# R/