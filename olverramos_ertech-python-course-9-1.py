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

raíz = A25
raíz.set_left(A14)
A25.set_right(A36)
A14.set_left(A9)
A14.set_right(A19)
A36.set_left(A29)
A36.set_right(A39)
A9.set_right(A10)
A19.set_right(A23)
A29.set_left(A27)
A23.set_left(A20)

# R/ 
# R/
# R/
# R/
# R/
# R/
# R/
# R/
# R/
# R/