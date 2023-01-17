# Print the Docker Image build date and git hash (https://github.com/Kaggle/docker-python)

!cat /etc/build_date

!cat /etc/git_commit
!pip show scipy numpy statsmodels
!pip install --upgrade scipy
import scipy

print(scipy.__version__)