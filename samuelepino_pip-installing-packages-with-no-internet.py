# Let's see what we have in our imported directory

! ls ../input/facenet_pytorch
!pip install facenet-pytorch --no-index --find-links=file:///kaggle/input/facenet_pytorch/ 
import facenet_pytorch



print("facenet_pytorch package successfully imported!")