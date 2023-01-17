!apt -o APT::Sandbox::User=root update
!apt -y install lsb-core
!sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
!apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
!apt -o APT::Sandbox::User=root update
!apt install -y ros-melodic-desktop-full
!rosdep init
!rosdep update
!echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
!bash -c 'source ~/.bashrc'
!apt install -y python-rosinstall python-rosinstall-generator python-wstool build-essential