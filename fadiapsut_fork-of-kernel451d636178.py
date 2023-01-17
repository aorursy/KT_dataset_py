# Python3 program to find the in and out degrees of the vertices of the given graph 

# Function to print the in and out degrees of all the vertices of the given graph 

def findInOutDegree(adjList, n): #no. of V is n+1

	_in = [0] * n 

	out = [0] * n 



	for i in range(0, len(adjList)): 

		List = adjList[i] 



		# Out degree for ith vertex will be the count of direct paths from i to other vertices 

		out[i] = len(List) 

		for j in range(0, len(List)): 



			# Every vertex that has an incoming edge from i 

			_in[List[j]] += 1



	print("Vertex\tIn\tOut") 

	for k in range(0, n): 

		print(str(k) + "\t" + str(_in[k]) +

					"\t" + str(out[k])) 



# Driver code 

if __name__ == "__main__": 

	# Adjacency list representation of the graph 

	adjList = [] 



	# from vertex 0 

	adjList.append([2,3]) 



	# Vertex 0 has an incoming edge from vertex 1 

	adjList.append([0]) 



	# from vertex 2 

	adjList.append([1,3,4]) 



	# from vertex 3 

	adjList.append([4,2]) 



	# from vertex 4 

	adjList.append([]) 



	n = len(adjList) 

	findInOutDegree(adjList, n) 
# Python program to detect cycle in a graph 

from collections import defaultdict 



class Graph(): 

	def __init__(self,vertices): 

		self.graph = defaultdict(list) 

		self.V = vertices 



	def addEdge(self,u,v): 

		self.graph[u].append(v) 



	def isCyclicUtil(self, v, visited, recStack): 



		# Mark current node as visited and adds to recursion stack 

		visited[v] = True

		recStack[v] = True



		# Recur for all neighbours if any neighbour is visited and in recStack then graph is cyclic 

		for neighbour in self.graph[v]: 

			if visited[neighbour] == False: 

				if self.isCyclicUtil(neighbour, visited, recStack) == True: 

					return True

			elif recStack[neighbour] == True: 

				return True



		# The node needs to be poped from recursion stack before function ends 

		recStack[v] = False

		return False



	# Returns true if graph is cyclic else false 

	def isCyclic(self): 

		visited = [False] * self.V 

		recStack = [False] * self.V 

		for node in range(self.V): 

			if visited[node] == False: 

				if self.isCyclicUtil(node,visited,recStack) == True: 

					return True

		return False



g = Graph(5) 

g.addEdge(1, 0) 

g.addEdge(0, 2) 

g.addEdge(2, 1) 

g.addEdge(0, 3) 

g.addEdge(3, 4)

g.addEdge(2, 4) 

g.addEdge(3, 2) 

g.addEdge(2, 3) 



if g.isCyclic() == 1: 

	print ("Graph has a cycle")

else: 

	print ("Graph has no cycle")
# Python implementation of Kosaraju's algorithm to print all SCCs 

from collections import defaultdict 



#This class represents a directed graph using adjacency list representation 

class Graph: 



	def __init__(self,vertices): 

		self.V= vertices #No. of vertices 

		self.graph = defaultdict(list) # default dictionary to store graph 



	# function to add an edge to graph 

	def addEdge(self,u,v): 

		self.graph[u].append(v) 



	# A function used by DFS 

	def DFSUtil(self,v,visited): 

		# Mark the current node as visited and print it 

		visited[v]= True

		print(v, end=" ")

		#Recur for all the vertices adjacent to this vertex 

		for i in self.graph[v]: 

			if visited[i]==False: 

				self.DFSUtil(i,visited) 



	def fillOrder(self,v,visited, stack): 

		# Mark the current node as visited 

		visited[v]= True

		#Recur for all the vertices adjacent to this vertex 

		for i in self.graph[v]: 

			if visited[i]==False: 

				self.fillOrder(i, visited, stack) 

		stack = stack.append(v) 



	# Function that returns reverse (or transpose) of this graph 

	def getTranspose(self): 

		g = Graph(self.V) 



		# Recur for all the vertices adjacent to this vertex 

		for i in self.graph: 

			for j in self.graph[i]: 

				g.addEdge(j,i) 

		return g 



	# The main function that finds and prints all strongly connected components 

	def printSCCs(self): 

		stack = [] 

		# Mark all the vertices as not visited (For first DFS) 

		visited =[False]*(self.V) 

		# Fill vertices in stack according to their finishing times 

		for i in range(self.V): 

			if visited[i]==False: 

				self.fillOrder(i, visited, stack) 



		# Create a reversed graph 

		gr = self.getTranspose() 

		# Mark all the vertices as not visited (For second DFS) 

		visited =[False]*(self.V) 



		# Now process all vertices in order defined by Stack 

		while stack: 

			i = stack.pop() 

			if visited[i]==False: 

				gr.DFSUtil(i, visited) 

				print("\n")



g = Graph(5) 

g.addEdge(1, 0) 

g.addEdge(0, 2) 

g.addEdge(2, 1) 

g.addEdge(0, 3) 

g.addEdge(3, 4)

g.addEdge(2, 4) 

g.addEdge(3, 2) 

g.addEdge(2, 3) 

  

print ("Following are strongly connected components in given graph") 

g.printSCCs() 
# program to check if there is exist a path between two vertices of a graph 

from collections import defaultdict 



#This class represents a directed graph using adjacency list representation 

class Graph: 



	def __init__(self,vertices): 

		self.V= vertices #No. of vertices 

		self.graph = defaultdict(list) # default dictionary to store graph 



	# function to add an edge to graph 

	def addEdge(self,u,v): 

		self.graph[u].append(v) 

	

	# Use BFS to check path between s and d 

	def isReachable(self, s, d): 

		# Mark all the vertices as not visited 

		visited =[False]*(self.V) 



		# Create a queue for BFS 

		queue=[] 



		# Mark the source node as visited and enqueue it 

		queue.append(s) 

		visited[s] = True



		while queue: 

			#Dequeue a vertex from queue 

			n = queue.pop(0) 

			# If this adjacent node is the destination node, then return true 

			if n == d: 

				return True



			# Else, continue to do BFS 

			for i in self.graph[n]: 

				if visited[i] == False: 

					queue.append(i) 

					visited[i] = True

		# If BFS is complete without visited d 

		return False



# Create a graph given in the above diagram 

g = Graph(5) 

g.addEdge(1, 0) 

g.addEdge(0, 2) 

g.addEdge(2, 1) 

g.addEdge(0, 3) 

g.addEdge(3, 4)

g.addEdge(2, 4) 

g.addEdge(3, 2) 

g.addEdge(2, 3) 



u =4; v =1

if g.isReachable(u, v): 

	print("There is a path from %d to %d" % (u,v)) 

else : 

	print("There is no path from %d to %d" % (u,v)) 



u =1; v =4

if g.isReachable(u, v) : 

	print("There is a path from %d to %d" % (u,v)) 

else : 

	print("There is no path from %d to %d" % (u,v)) 

    

u =0; v =3

if g.isReachable(u, v): 

	print("There is a path from %d to %d" % (u,v)) 

else : 

	print("There is no path from %d to %d" % (u,v)) 



u =3; v =0

if g.isReachable(u, v): 

	print("There is a path from %d to %d" % (u,v)) 

else : 

	print("There is no path from %d to %d" % (u,v)) 