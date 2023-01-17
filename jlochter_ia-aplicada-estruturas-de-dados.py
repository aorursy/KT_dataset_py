class Node:
    def __init__(self,val):
        self.val = val
        self.next = None
        
    def traverse(self):
        node = self
        while node != None:
            print(node.val)
            node = node.next
            
    def removeDuplicates(self):
        elements = []
        previous = None
        node = self
        while node != None:
            if node.val in elements:
                previous.next = node.next
                node.next = None
                node = previous
            else:
                elements.append(node.val)
            previous = node
            node = node.next
            
    def get_kth_element(self,k):
        node = self
        i = 0
        while node != None and i < k:
            node = node.next
            i += 1
        if node != None and i == k:
            print(node.val)
        else:
            print('not found')     
                  
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(1)
node1.next = node2
node2.next = node3
node3.next = node4
print('Navegando entre os nós:')
node1.traverse()
print()
print('Removendo os duplicados...')
node1.removeDuplicates()
print()
print('Verificando se foram removidos:')
node1.traverse()
print()
print('Encontrando o elemento da lista baseado num índice:')
node1.get_kth_element(0)
print()